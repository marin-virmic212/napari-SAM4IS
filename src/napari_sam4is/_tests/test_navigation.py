import numpy as np
import pytest
from napari.utils.key_bindings import KeyBinding

from napari_sam4is import SAMWidget
from napari_sam4is._navigation import (
    _DIRECTIONS,
    cyclic_step,
    nearest_in_direction,
    next_in_order,
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
        (0, +1, 1),
        (2, +1, 0),  # wraps forward
        (0, -1, 2),  # wraps backward
        (None, +1, 0),
        (None, -1, 2),
    ],
)
def test_cyclic_step(current, step, expected):
    assert cyclic_step([0, 1, 2], current, step) == expected


def test_cyclic_step_empty():
    assert cyclic_step([], 0, +1) is None


@pytest.mark.parametrize(
    "current, step, expected",
    [
        (2, +1, 3),  # current itself is not in the subset
        (5, +1, 1),  # wraps forward
        (2, -1, 1),
        (0, -1, 3),  # wraps backward
        (None, +1, 1),
    ],
)
def test_next_in_order(current, step, expected):
    assert next_in_order([1, 3], current, step) == expected


def test_nearest_in_direction_prefers_aligned_shape():
    centers = {
        0: np.array([50.0, 50.0]),
        1: np.array([50.0, 80.0]),  # straight right, farther
        2: np.array([20.0, 65.0]),  # closer along x but far off-axis
        3: np.array([50.0, 10.0]),  # to the left
    }
    assert nearest_in_direction(centers, 0, _DIRECTIONS["Right"]) == 1
    assert nearest_in_direction(centers, 0, _DIRECTIONS["Left"]) == 3
    assert nearest_in_direction(centers, 0, _DIRECTIONS["Up"]) == 2
    assert nearest_in_direction(centers, 0, _DIRECTIONS["Down"]) is None


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
    return widget, layer, viewer


def test_navigation_keys_bound_on_output_layer(nav_widget):
    widget, layer, _ = nav_widget
    for key in ("]", "[", "N", "Shift-N", "Shift-Right", "Alt-B", "Alt-A"):
        assert KeyBinding.from_str(key) in layer.keymap
    widget._disconnect_output_layer_events()
    assert KeyBinding.from_str("]") not in layer.keymap


def test_step_selects_and_centers(nav_widget):
    widget, layer, _ = nav_widget
    nav = widget._navigator

    assert nav.step(+1) == 0
    assert layer.selected_data == {0}
    np.testing.assert_allclose(nav._camera().center[-2:], [15, 15])

    assert nav.step(+1) == 1
    assert nav.step(+1) == 2
    assert nav.step(+1) == 0  # wraps
    assert nav.step(-1) == 2


def test_step_direction_and_go_back(nav_widget):
    widget, layer, _ = nav_widget
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
    widget, layer, _ = nav_widget
    nav = widget._navigator
    widget._ensure_features_columns(layer)

    layer.selected_data = {1}
    widget._accept_selected()
    layer.selected_data = set()

    assert nav.step_unreviewed(+1) == 0
    assert nav.accept_and_next() == 2  # 1 is already accepted
    assert layer.features["review_status"].iloc[0] == "approved"

    assert nav.accept_and_next() is None
    assert set(layer.features["review_status"]) == {"approved"}


def test_go_back_skips_deleted_annotation(nav_widget):
    widget, layer, _ = nav_widget
    nav = widget._navigator

    nav.step(+1)
    nav.step(+1)
    nav.step(+1)  # history: [0, 1]
    layer.selected_data = {0, 1}
    layer.remove_selected()  # only annotation 0 remains
    layer.selected_data = {0}

    assert nav.go_back() == 0
