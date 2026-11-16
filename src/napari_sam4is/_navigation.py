"""Keyboard navigation between annotations in the output Shapes layer.

Lets the reviewer step through annotations one by one and centers the
view on each of them:

- ``]`` / ``[``: next / previous annotation, in drawing order
- ``N`` / ``Shift-N``: next / previous annotation not yet accepted
- ``Shift-Up/Down/Left/Right``: nearest annotation in that direction
- ``Alt-B``: back to the previously selected annotation
- ``Alt-A``: accept the selected annotation and jump to the next
  annotation not yet accepted
"""

import contextlib

import numpy as np

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

_NAV_KEYS = (
    "]",
    "[",
    "N",
    "Shift-N",
    *(f"Shift-{name}" for name in _DIRECTIONS),
    "Alt-B",
    "Alt-A",
)


def shape_center(layer, idx):
    """Return the (y, x) center of a shape's bounding box.

    The bounding box is used rather than the vertex mean so that shapes
    with holes, whose vertices also include the hole rings, are centered
    on their outer boundary.
    """
    yx = np.asarray(layer.data[idx])[:, -2:]
    return (yx.min(axis=0) + yx.max(axis=0)) / 2


def is_unreviewed(layer, idx):
    """True when the annotation has not been accepted yet."""
    features = layer.features
    if "review_status" not in features.columns or idx >= len(features):
        return True
    return features["review_status"].iloc[idx] != "approved"


def cyclic_step(indices, current, step):
    """Return the entry ``step`` positions away from ``current``.

    Wraps around at both ends. If ``current`` is not in ``indices``,
    returns the first entry for a forward step and the last one for a
    backward step. Returns None for an empty list.
    """
    if not indices:
        return None
    if current in indices:
        pos = (indices.index(current) + step) % len(indices)
        return indices[pos]
    return indices[0] if step > 0 else indices[-1]


def next_in_order(indices, current, step):
    """Like :func:`cyclic_step` for a subset that may exclude ``current``.

    Picks the first entry after ``current`` in drawing order (or before
    it for a negative step), wrapping around.
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
        centers (dict): index -> (y, x) center
        current (int): index of the selected annotation
        direction (np.ndarray): unit (dy, dx) vector

    :return: index of the best candidate, or None if nothing lies in
        that direction
    """
    here = centers[current]
    best_idx, best_score = None, None
    for idx, center in centers.items():
        if idx == current:
            continue
        offset = np.asarray(center) - here
        along = float(offset @ direction)
        if along <= 0:
            continue
        perp = abs(float(offset[0] * direction[1] - offset[1] * direction[0]))
        score = along + _PERPENDICULAR_WEIGHT * perp
        if best_score is None or score < best_score:
            best_idx, best_score = idx, score
    return best_idx


class AnnotationNavigator:
    """Move the selection between annotations of one Shapes layer."""

    def __init__(self, viewer, accept_selected=None):
        """
        Args:
            viewer: the napari viewer
            accept_selected (callable): marks the selected annotations as
                accepted; required for the accept-and-next key
        """
        self._viewer = viewer
        self._accept_selected = accept_selected
        self._history = []
        self._layer = None

    # --- key binding ---

    def connect(self, layer):
        """Bind the navigation keys on ``layer``."""
        self.disconnect()
        bindings = {
            "]": lambda _: self.step(+1),
            "[": lambda _: self.step(-1),
            "N": lambda _: self.step_unreviewed(+1),
            "Shift-N": lambda _: self.step_unreviewed(-1),
            "Alt-B": lambda _: self.go_back(),
            "Alt-A": lambda _: self.accept_and_next(),
        }
        for name in _DIRECTIONS:
            bindings[f"Shift-{name}"] = (
                lambda _, name=name: self.step_direction(name)
            )
        for key, func in bindings.items():
            layer.bind_key(key, func, overwrite=True)
        self._layer = layer

    def disconnect(self):
        """Remove the navigation keys from the connected layer."""
        if self._layer is None:
            return
        for key in _NAV_KEYS:
            with contextlib.suppress(TypeError, RuntimeError, KeyError):
                self._layer.bind_key(key, None)
        self._layer = None
        self._history.clear()

    # --- actions ---

    def step(self, step):
        """Select the next / previous annotation in drawing order."""
        layer = self._layer
        if layer is None or len(layer.data) == 0:
            self._set_status("No annotations")
            return None
        indices = list(range(len(layer.data)))
        idx = cyclic_step(indices, self._current(), step)
        self._select(idx, f"Annotation {idx + 1} / {len(indices)}")
        return idx

    def step_unreviewed(self, step):
        """Select the next / previous annotation not yet accepted."""
        layer = self._layer
        if layer is None:
            return None
        pending = [
            i for i in range(len(layer.data)) if is_unreviewed(layer, i)
        ]
        if not pending:
            self._set_status("All annotations are accepted")
            return None
        idx = next_in_order(pending, self._current(), step)
        self._select(
            idx,
            f"Unreviewed {pending.index(idx) + 1} / {len(pending)}"
            f" (annotation {idx + 1})",
        )
        return idx

    def step_direction(self, name):
        """Select the nearest annotation in direction ``name``."""
        layer = self._layer
        current = self._current()
        if layer is None or current is None:
            self._set_status("Select an annotation first")
            return None
        centers = {i: shape_center(layer, i) for i in range(len(layer.data))}
        idx = nearest_in_direction(centers, current, _DIRECTIONS[name])
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
        self._camera().center = tuple(shape_center(layer, idx))
        self._set_status(status)

    def _camera(self):
        # napari 0.9 moved the camera to viewer.scene.camera
        scene = getattr(self._viewer, "scene", None)
        return scene.camera if scene is not None else self._viewer.camera

    def _set_status(self, text):
        self._viewer.status = text
