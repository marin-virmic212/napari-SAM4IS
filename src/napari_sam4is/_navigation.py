"""Keyboard navigation between annotations in the output Shapes layer.

Lets the reviewer step through annotations one by one and centers the
view on each of them:

- ``]`` / ``[``: next / previous annotation, in drawing order
- ``N`` / ``Shift-N``: next / previous annotation not yet accepted
- ``Q`` / ``Shift-Q``: next / previous annotation marked as unclear
  boundary or uncertain class
- ``Shift-Up/Down/Left/Right``: nearest annotation in that direction
- ``Alt-B``: back to the previously selected annotation
- ``Alt-A``: accept the selected annotation and jump to the next
  annotation not yet accepted

The :class:`NavigationPanel` widget adds options to limit navigation to
one class (and its subclasses) and to zoom to each annotation, and shows
the review progress of the output layer.
"""

import contextlib

import numpy as np
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
)

# Image coordinates are (y, x); y grows downwards.
_DIRECTIONS = {
    "Up": np.array([-1.0, 0.0]),
    "Down": np.array([1.0, 0.0]),
    "Left": np.array([0.0, -1.0]),
    "Right": np.array([0.0, 1.0]),
}
# How strongly sideways offset is penalized when picking a neighbor.
_PERPENDICULAR_WEIGHT = 2.0
_HISTORY_LIMIT = 100
# Share of the canvas the annotation fills when zooming to it.
_ZOOM_FILL = 0.5
# Do not zoom in further than this on tiny annotations.
_MAX_ZOOM = 20.0

_NAV_KEYS = (
    "]",
    "[",
    "N",
    "Shift-N",
    "Q",
    "Shift-Q",
    *(f"Shift-{name}" for name in _DIRECTIONS),
    "Alt-B",
    "Alt-A",
)
_LAYER_EVENTS = ("data", "features", "highlight")

_KEY_HELP = (
    "] / [ : next / previous\n"
    "N / Shift+N : next / previous not accepted\n"
    "Q / Shift+Q : next / previous unclear or uncertain\n"
    "Alt+A : accept and go to next not accepted\n"
    "Shift+Arrow : nearest in that direction\n"
    "Alt+B : back to previous selection"
)


def shape_bounds(layer, idx):
    """Return the (y, x) min and max corners of a shape's bounding box."""
    yx = np.asarray(layer.data[idx])[:, -2:]
    return yx.min(axis=0), yx.max(axis=0)


def shape_center(layer, idx):
    """Return the (y, x) center of a shape's bounding box.

    The bounding box is used rather than the vertex mean so that shapes
    with holes, whose vertices also include the hole rings, are centered
    on their outer boundary.
    """
    low, high = shape_bounds(layer, idx)
    return (low + high) / 2


def _feature(layer, column, idx, default):
    features = layer.features
    if column not in features.columns or idx >= len(features):
        return default
    return features[column].iloc[idx]


def is_unreviewed(layer, idx):
    """True when the annotation has not been accepted yet."""
    return _feature(layer, "review_status", idx, "unreviewed") != "approved"


def is_flagged(layer, idx):
    """True when the annotation is marked unclear or uncertain."""
    return bool(_feature(layer, "unclear", idx, False)) or bool(
        _feature(layer, "uncertain", idx, False)
    )


def review_counts(layer):
    """Return (accepted, flagged, total) counts for the layer."""
    total = len(layer.data)
    accepted = sum(not is_unreviewed(layer, i) for i in range(total))
    flagged = sum(is_flagged(layer, i) for i in range(total))
    return accepted, flagged, total


def next_in_order(indices, current, step):
    """Return the entry after (or before) ``current`` in ``indices``.

    ``indices`` must be sorted. ``current`` does not need to be one of
    them. Wraps around at both ends. With no current selection, returns
    the first entry for a forward step and the last one for a backward
    step. Returns None for an empty list.
    """
    if not indices:
        return None
    if current is None:
        return indices[0] if step > 0 else indices[-1]
    if step > 0:
        after = [i for i in indices if i > current]
        return after[0] if after else indices[0]
    before = [i for i in indices if i < current]
    return before[-1] if before else indices[-1]


def nearest_in_direction(centers, current, direction):
    """Pick the annotation closest to ``current`` in ``direction``.

    Args:
        centers (dict): index -> (y, x) center of each candidate
        current (np.ndarray): (y, x) center of the selected annotation
        direction (np.ndarray): unit (dy, dx) vector

    :return: index of the best candidate, or None if nothing lies in
        that direction
    """
    best_idx, best_score = None, None
    for idx, center in centers.items():
        offset = np.asarray(center) - current
        along = float(offset @ direction)
        if along <= 0:
            continue
        perp = abs(float(offset[0] * direction[1] - offset[1] * direction[0]))
        score = along + _PERPENDICULAR_WEIGHT * perp
        if best_score is None or score < best_score:
            best_idx, best_score = idx, score
    return best_idx


def zoom_to_fit(bounds_size, canvas_size):
    """Camera zoom at which a box fills ``_ZOOM_FILL`` of the canvas.

    Args:
        bounds_size: (height, width) of the box in data pixels
        canvas_size: (height, width) of the canvas in screen pixels
    """
    size = np.maximum(np.asarray(bounds_size, dtype=float), 1.0)
    zoom = float(np.min(np.asarray(canvas_size) * _ZOOM_FILL / size))
    return min(zoom, _MAX_ZOOM)


class AnnotationNavigator:
    """Move the selection between annotations of one Shapes layer."""

    def __init__(self, viewer, accept_selected=None, class_filter=None):
        """
        Args:
            viewer: the napari viewer
            accept_selected (callable): marks the selected annotations as
                accepted; required for the accept-and-next key
            class_filter (callable): returns the set of class strings to
                navigate between, or None to navigate between all
        """
        self._viewer = viewer
        self._accept_selected = accept_selected
        self._class_filter = class_filter
        self.zoom_to_annotation = False
        self._history = []
        self._layer = None
        self._listeners = []

    @property
    def layer(self):
        return self._layer

    def add_listener(self, callback):
        """Call ``callback()`` when the connected layer or its data change."""
        self._listeners.append(callback)

    # --- key binding ---

    def connect(self, layer):
        """Bind the navigation keys on ``layer``."""
        self.disconnect()
        bindings = {
            "]": lambda _: self.step(+1),
            "[": lambda _: self.step(-1),
            "N": lambda _: self.step_unreviewed(+1),
            "Shift-N": lambda _: self.step_unreviewed(-1),
            "Q": lambda _: self.step_flagged(+1),
            "Shift-Q": lambda _: self.step_flagged(-1),
            "Alt-B": lambda _: self.go_back(),
            "Alt-A": lambda _: self.accept_and_next(),
        }
        for name in _DIRECTIONS:
            bindings[f"Shift-{name}"] = (
                lambda _, name=name: self.step_direction(name)
            )
        for key, func in bindings.items():
            layer.bind_key(key, func, overwrite=True)
        for name in _LAYER_EVENTS:
            getattr(layer.events, name).connect(self._notify)
        self._layer = layer
        self._notify()

    def disconnect(self):
        """Remove the navigation keys from the connected layer."""
        if self._layer is None:
            return
        for key in _NAV_KEYS:
            with contextlib.suppress(TypeError, RuntimeError, KeyError):
                self._layer.bind_key(key, None)
        for name in _LAYER_EVENTS:
            with contextlib.suppress(TypeError, RuntimeError):
                getattr(self._layer.events, name).disconnect(self._notify)
        self._layer = None
        self._history.clear()
        self._notify()

    def _notify(self, event=None):
        for callback in self._listeners:
            callback()

    # --- actions ---

    def step(self, step):
        """Select the next / previous annotation in drawing order."""
        return self._step_matching(step, None, "Annotation", "No annotations")

    def step_unreviewed(self, step):
        """Select the next / previous annotation not yet accepted."""
        return self._step_matching(
            step,
            is_unreviewed,
            "Not accepted",
            "All annotations are accepted",
        )

    def step_flagged(self, step):
        """Select the next / previous unclear or uncertain annotation."""
        return self._step_matching(
            step,
            is_flagged,
            "Unclear/uncertain",
            "No unclear or uncertain annotations",
        )

    def step_direction(self, name):
        """Select the nearest annotation in direction ``name``."""
        layer = self._layer
        current = self._current()
        if layer is None or current is None:
            self._set_status("Select an annotation first")
            return None
        centers = {
            i: shape_center(layer, i)
            for i in self._candidates()
            if i != current
        }
        idx = nearest_in_direction(
            centers, shape_center(layer, current), _DIRECTIONS[name]
        )
        if idx is None:
            self._set_status(f"No annotation to the {name.lower()}")
            return None
        self._select(idx, f"Annotation {idx + 1} / {len(layer.data)}")
        return idx

    def go_back(self):
        """Return to the previously selected annotation."""
        layer = self._layer
        while self._history:
            idx = self._history.pop()
            # skip entries whose annotation was deleted since
            if layer is not None and idx < len(layer.data):
                self._select(
                    idx,
                    f"Back to annotation {idx + 1}"
                    f" ({len(self._history)} left in history)",
                    record=False,
                )
                return idx
        self._set_status("No history to go back to")
        return None

    def accept_and_next(self):
        """Accept the selection, then jump to the next unreviewed one."""
        if self._layer is None or self._accept_selected is None:
            return None
        if not self._layer.selected_data:
            self._set_status("Select an annotation first")
            return None
        self._accept_selected()
        return self.step_unreviewed(+1)

    # --- helpers ---

    def _active_classes(self):
        """Allowed class strings, or None when navigation is unfiltered."""
        return self._class_filter() if self._class_filter else None

    def _candidates(self, predicate=None):
        """Sorted indices passing the class filter and ``predicate``."""
        layer = self._layer
        classes = self._active_classes()
        return [
            i
            for i in range(len(layer.data))
            if (classes is None or _feature(layer, "class", i, "") in classes)
            and (predicate is None or predicate(layer, i))
        ]

    def _step_matching(self, step, predicate, label, empty_status):
        if self._layer is None:
            return None
        classes = self._active_classes()
        if classes is not None and not classes:
            self._set_status("Select a class in the class list first")
            return None
        matches = self._candidates(predicate)
        if not matches:
            if classes is not None:
                empty_status += " in the selected class"
            self._set_status(empty_status)
            return None
        idx = next_in_order(matches, self._current(), step)
        status = f"{label} {matches.index(idx) + 1} / {len(matches)}"
        if predicate is not None or classes is not None:
            status += f" (annotation {idx + 1})"
        self._select(idx, status)
        return idx

    def _current(self):
        """Index of the single selected annotation, else None."""
        selected = list(self._layer.selected_data)
        return selected[0] if len(selected) == 1 else None

    def _select(self, idx, status, record=True):
        layer = self._layer
        previous = self._current()
        if record and previous is not None and previous != idx:
            self._history.append(previous)
            del self._history[:-_HISTORY_LIMIT]
        layer.mode = "select"
        layer.selected_data = {idx}
        camera = self._camera()
        low, high = shape_bounds(layer, idx)
        camera.center = tuple((low + high) / 2)
        canvas_size = self._canvas_size()
        if self.zoom_to_annotation and canvas_size is not None:
            camera.zoom = zoom_to_fit(high - low, canvas_size)
        self._set_status(status)

    def _camera(self):
        # napari 0.9 moved the camera to viewer.scene.camera
        scene = getattr(self._viewer, "scene", None)
        return scene.camera if scene is not None else self._viewer.camera

    def _canvas_size(self):
        """Canvas (height, width) in screen pixels, if napari exposes it."""
        canvas = getattr(self._viewer, "canvas", None)
        if canvas is not None and hasattr(canvas, "size"):
            return canvas.size
        return getattr(self._viewer, "_canvas_size", None)

    def _set_status(self, text):
        self._viewer.status = text


class NavigationPanel(QGroupBox):
    """Navigation options and review progress for the SAM widget."""

    def __init__(self, sam_widget):
        super().__init__("Navigation")
        self._sam_widget = sam_widget
        self.navigator = AnnotationNavigator(
            sam_widget._viewer,
            accept_selected=sam_widget._accept_selected,
            class_filter=self._selected_classes,
        )

        layout = QVBoxLayout()
        self._progress_label = QLabel()
        layout.addWidget(self._progress_label)

        class_row = QHBoxLayout()
        class_row.addWidget(QLabel("Class:"))
        self.class_combo = _ClassFilterCombo(self._class_choices)
        self.class_combo.setToolTip(
            "Move only between annotations of this class\n"
            "(subclasses included)"
        )
        class_row.addWidget(self.class_combo, 1)
        layout.addLayout(class_row)

        self.zoom_checkbox = QCheckBox("Zoom to annotation")
        self.zoom_checkbox.setToolTip(
            "Also zoom so that the annotation fills about half the view"
        )
        self.zoom_checkbox.toggled.connect(self._on_zoom_toggled)
        layout.addWidget(self.zoom_checkbox)

        help_label = QLabel(_KEY_HELP)
        help_label.setStyleSheet("color: gray;")
        layout.addWidget(help_label)
        self.setLayout(layout)

        self.navigator.add_listener(self.update_progress)
        self.update_progress()

    def _on_zoom_toggled(self, checked):
        self.navigator.zoom_to_annotation = checked

    def _class_choices(self):
        """Class strings currently in the class tree, in tree order."""
        widget = self._sam_widget
        return [
            widget._get_item_class_string(item)
            for item in widget._iter_all_tree_items()
        ]

    def _selected_classes(self):
        """The chosen class string and those of its subclasses.

        Returns None when navigating between all classes. The class tree
        is read on every call so that renamed or added classes count.
        """
        chosen = self.class_combo.currentData()
        if chosen is None:
            return None
        widget = self._sam_widget
        for item in widget._iter_all_tree_items():
            if widget._get_item_class_string(item) == chosen:
                classes = set()
                stack = [item]
                while stack:
                    node = stack.pop()
                    classes.add(widget._get_item_class_string(node))
                    stack.extend(
                        node.child(i) for i in range(node.childCount())
                    )
                return classes
        # the chosen class no longer exists in the tree
        return {chosen}

    def update_progress(self, *_):
        layer = self.navigator.layer
        if layer is None:
            self._progress_label.setText("No output shapes layer")
            return
        accepted, flagged, total = review_counts(layer)
        self._progress_label.setText(
            f"Accepted {accepted} / {total}"
            f"  ·  Unclear/uncertain {flagged}"
        )


class _ClassFilterCombo(QComboBox):
    """Class choice that refreshes from the class tree when opened."""

    ALL = "All classes"

    def __init__(self, get_choices):
        super().__init__()
        self._get_choices = get_choices
        self.addItem(self.ALL, None)

    def refresh(self):
        current = self.currentData()
        self.blockSignals(True)
        self.clear()
        self.addItem(self.ALL, None)
        for class_str in self._get_choices():
            self.addItem(class_str, class_str)
        index = self.findData(current) if current is not None else 0
        self.setCurrentIndex(max(index, 0))
        self.blockSignals(False)

    def showPopup(self):
        self.refresh()
        super().showPopup()
