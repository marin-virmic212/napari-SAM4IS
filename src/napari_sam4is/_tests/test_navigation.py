import numpy as np
import pytest
from napari.utils.key_bindings import KeyBinding
from qtpy.QtWidgets import QTreeWidgetItem

from napari_sam4is import SAMWidget
from napari_sam4is._navigation import (
    _DIRECTIONS,
    _MAX_ZOOM,
    nearest_in_direction,
    next_in_order,
    zoom_to_fit,
)


def _square(y, x, size=10):
    return np.array(
        [[y, x], [y, x + size], [y + size, x + size], [y + size, x]],
        dtype=float,
    )


# --- pure helpers ---


@pytest.mark.parametrize(
    "current, step, expected",
    [
        (1, +1, 3),
        (2, +1, 3),  # current itself is not in the list
        (5, +1, 1),  # wraps forward
        (3, -1, 1),
        (0, -1, 5),  # wraps backward
        (None, +1, 1),
        (None, -1, 5),
    ],
)
def test_next_in_order(current, step, expected):
    assert next_in_order([1, 3, 5], current, step) == expected


def test_next_in_order_empty():
    assert next_in_order([], 0, +1) is None


def test_nearest_in_direction_prefers_aligned_shape():
    here = np.array([50.0, 50.0])
    centers = {
        1: np.array([50.0, 80.0]),  # straight right, farther
        2: np.array([20.0, 65.0]),  # closer along x but far off-axis
        3: np.array([50.0, 10.0]),  # to the left
    }
    assert nearest_in_direction(centers, here, _DIRECTIONS["Right"]) == 1
    assert nearest_in_direction(centers, here, _DIRECTIONS["Left"]) == 3
    assert nearest_in_direction(centers, here, _DIRECTIONS["Up"]) == 2
    assert nearest_in_direction(centers, here, _DIRECTIONS["Down"]) is None


def test_zoom_to_fit():
    # a 100 x 50 box on a 600 x 800 canvas: height limits, fills half
    assert zoom_to_fit((100, 50), (600, 800)) == pytest.approx(3.0)
    # tiny or degenerate boxes are capped
    assert zoom_to_fit((0, 0), (600, 800)) == _MAX_ZOOM


# --- widget integration ---


@pytest.fixture
def nav_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    viewer.add_image(np.random.random((200, 200)))
    widget = SAMWidget(viewer)
    layer = widget._accepted_layer
    layer.add_polygons([_square(10, 10), _square(10, 100), _square(100, 10)])
    widget._shapes_layer_selection.setCurrentText(layer.name)
    widget._connect_output_layer_events(layer)
    widget._ensure_features_columns(layer)
    return widget, layer


def _set_feature(layer, column, values):
    features = layer.features.copy()
    features[column] = values
    layer.features = features


def test_navigation_keys_bound_on_output_layer(nav_widget):
    widget, layer = nav_widget
    keys = ("]", "[", "N", "Shift-N", "Q", "Shift-Q", "Shift-Right")
    for key in (*keys, "Alt-B", "Alt-A"):
        assert KeyBinding.from_str(key) in layer.keymap
    widget._disconnect_output_layer_events()
    assert KeyBinding.from_str("]") not in layer.keymap


def test_step_selects_and_centers(nav_widget):
    widget, layer = nav_widget
    nav = widget._navigator

    assert nav.step(+1) == 0
    assert layer.selected_data == {0}
    np.testing.assert_allclose(nav._camera().center[-2:], [15, 15])

    assert nav.step(+1) == 1
    assert nav.step(+1) == 2
    assert nav.step(+1) == 0  # wraps
    assert nav.step(-1) == 2


def test_step_direction_and_go_back(nav_widget):
    widget, layer = nav_widget
    nav = widget._navigator

    nav.step(+1)  # annotation 0, top-left
    assert nav.step_direction("Right") == 1
    assert nav.step_direction("Right") is None
    assert layer.selected_data == {1}

    assert nav.go_back() == 0
    assert layer.selected_data == {0}
    assert nav.step_direction("Down") == 2
    assert nav.go_back() == 0
    assert nav.go_back() is None


def test_accept_and_next_skips_accepted(nav_widget):
    widget, layer = nav_widget
    nav = widget._navigator

    layer.selected_data = {1}
    widget._accept_selected()
    layer.selected_data = set()

    assert nav.step_unreviewed(+1) == 0
    assert nav.accept_and_next() == 2  # 1 is already accepted
    assert layer.features["review_status"].iloc[0] == "approved"

    assert nav.accept_and_next() is None
    assert set(layer.features["review_status"]) == {"approved"}


def test_step_flagged(nav_widget):
    widget, layer = nav_widget
    nav = widget._navigator

    assert nav.step_flagged(+1) is None
    _set_feature(layer, "unclear", [False, True, False])
    _set_feature(layer, "uncertain", [False, False, True])

    assert nav.step_flagged(+1) == 1
    assert nav.step_flagged(+1) == 2
    assert nav.step_flagged(+1) == 1  # wraps


def test_class_filter_includes_subclasses(nav_widget):
    widget, layer = nav_widget
    nav = widget._navigator
    panel = widget._navigation_panel

    animal = QTreeWidgetItem(["0: animal"])
    animal.addChild(QTreeWidgetItem(["1: cat"]))
    widget._class_list_widget.addTopLevelItem(animal)
    widget._class_list_widget.addTopLevelItem(QTreeWidgetItem(["2: tree"]))
    _set_feature(layer, "class", ["2: tree", "1: animal-cat", "0: animal"])

    panel.class_combo.refresh()
    assert panel.class_combo.count() == 4  # "All classes" + 3

    panel.class_combo.setCurrentIndex(panel.class_combo.findData("0: animal"))
    assert nav.step(+1) == 1
    assert nav.step(+1) == 2
    assert nav.step(+1) == 1  # annotation 0 (tree) is skipped

    panel.class_combo.setCurrentIndex(
        panel.class_combo.findData("1: animal-cat")
    )
    assert nav.step(+1) == 1
    assert nav.step(+1) == 1

    panel.class_combo.setCurrentIndex(0)  # all classes
    assert nav.step(+1) == 2


def test_zoom_to_annotation(nav_widget):
    widget, layer = nav_widget
    nav = widget._navigator
    camera = nav._camera()
    camera.zoom = 1.0

    nav.step(+1)
    assert camera.zoom == pytest.approx(1.0)

    widget._navigation_panel.zoom_checkbox.setChecked(True)
    nav.step(+1)
    expected = zoom_to_fit((10, 10), nav._canvas_size())
    assert camera.zoom == pytest.approx(expected)


def test_progress_label_updates(nav_widget):
    widget, layer = nav_widget
    label = widget._navigation_panel._progress_label

    assert label.text().startswith("Accepted 0 / 3")
    layer.selected_data = {0}
    widget._accept_selected()
    assert label.text().startswith("Accepted 1 / 3")
    widget._accept_all()
    assert label.text().startswith("Accepted 3 / 3")

    _set_feature(layer, "unclear", [True, False, False])
    assert label.text().endswith("Unclear/uncertain 1")

    widget._disconnect_output_layer_events()
    assert label.text() == "No output shapes layer"


def test_go_back_skips_deleted_annotation(nav_widget):
    widget, layer = nav_widget
    nav = widget._navigator

    nav.step(+1)
    nav.step(+1)
    nav.step(+1)  # history: [0, 1]
    layer.selected_data = {0, 1}
    layer.remove_selected()  # only annotation 0 remains
    layer.selected_data = {0}

    assert nav.go_back() == 0
