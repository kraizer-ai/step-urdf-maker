from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from urdf_maker.model import ScenePart
from urdf_maker.ui.viewport import ViewportWidget, _coplanar_region_geometry


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _part(identifier: str) -> ScenePart:
    return ScenePart(
        identifier,
        identifier,
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float),
        np.asarray(((0, 1, 2),), dtype=np.int64),
        color=(0.2, 0.3, 0.4, 1.0),
    )


def test_coplanar_pick_uses_connected_face_center_and_normal() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 1.0),
            (3.0, 0.0, 0.0),
            (4.0, 0.0, 0.0),
            (3.0, 1.0, 0.0),
        )
    )
    triangles = np.asarray(
        (
            (0, 1, 2),
            (0, 2, 3),
            (1, 4, 2),  # connected, but perpendicular to the picked plane
            (5, 6, 7),  # coplanar, but disconnected from the picked face
        ),
        dtype=np.int64,
    )

    center, normal, count = _coplanar_region_geometry(vertices, triangles, 0)

    np.testing.assert_allclose(center, (0.5, 0.5, 0.0))
    np.testing.assert_allclose(normal, (0.0, 0.0, 1.0))
    assert count == 2


def test_transient_part_colors_preserve_selection_outline() -> None:
    app = _app()
    viewport = ViewportWidget()
    try:
        viewport.set_parts([_part("one")])
        viewport.set_color_overrides({"one": (0.1, 0.8, 0.2, 1.0)})
        prop = viewport._parts["one"].actor.GetProperty()

        np.testing.assert_allclose(prop.GetColor(), (0.1, 0.8, 0.2), atol=1e-9)
        assert not prop.GetEdgeVisibility()

        viewport.set_selected(["one"])
        assert not prop.GetEdgeVisibility()
        np.testing.assert_allclose(prop.GetColor(), (0.1, 0.8, 0.2), atol=1e-9)
        assert "one" in viewport._selection_outlines
        outline = viewport._selection_outlines["one"].actor
        assert outline.GetProperty().GetLineWidth() >= 3.0
        assert viewport._selection_renderer.GetLayer() > viewport._renderer.GetLayer()
        assert not viewport._selection_renderer.GetPreserveDepthBuffer()

        transform = np.eye(4)
        transform[:3, 3] = (1.0, 2.0, 3.0)
        viewport.update_part_transforms({"one": transform})
        outline_matrix = outline.GetUserMatrix()
        np.testing.assert_allclose(
            [outline_matrix.GetElement(index, 3) for index in range(3)],
            (1.0, 2.0, 3.0),
        )

        viewport.set_selected([])
        np.testing.assert_allclose(prop.GetColor(), (0.1, 0.8, 0.2), atol=1e-9)
        assert not prop.GetEdgeVisibility()
        assert viewport._selection_outlines == {}
    finally:
        viewport._vtk_widget.Finalize()
        viewport.close()
        viewport.deleteLater()
        app.processEvents()


def test_surface_pick_result_includes_display_transform_and_pick_mode() -> None:
    app = _app()
    viewport = ViewportWidget()
    try:
        viewport.set_parts([_part("one")])
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(
            ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))
        )
        transform[:3, 3] = (2.0, 3.0, 4.0)
        viewport.update_part_transforms({"one": transform})

        picked = viewport._surface_pick_result("one", 0)
        np.testing.assert_allclose(picked["center_zero"], (1 / 3, 1 / 3, 0.0))
        np.testing.assert_allclose(picked["normal_zero"], (0.0, 0.0, 1.0))
        np.testing.assert_allclose(picked["center_world"], (2 + 1 / 3, 3.0, 4 + 1 / 3))
        np.testing.assert_allclose(picked["normal_world"], (0.0, -1.0, 0.0))

        viewport.begin_surface_pick()
        assert viewport.surface_pick_active()
        assert "평면을 클릭" in viewport._shortcut_label.text()
        viewport.cancel_surface_pick()
        assert not viewport.surface_pick_active()
        assert "Ctrl+H" in viewport._shortcut_label.text()
    finally:
        viewport._vtk_widget.Finalize()
        viewport.close()
        viewport.deleteLater()
        app.processEvents()


def test_viewport_shows_fps_overlay_and_uses_60_hz_update_hint() -> None:
    app = _app()
    viewport = ViewportWidget()
    try:
        assert viewport._fps_label.text() == "FPS: 대기"
        assert viewport._fps_label.testAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        assert "Ctrl+H" in viewport._shortcut_label.text()
        assert viewport._shortcut_label.testAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        assert viewport._render_window.GetDesiredUpdateRate() == 60.0
        assert viewport._interactor.GetDesiredUpdateRate() == 60.0

        viewport.show()
        viewport.resize(520, 320)
        for _ in range(4):
            app.processEvents()
        assert viewport._last_render_completed > 0.0
        assert viewport._frames_since_sample > 0

        toggles: list[bool] = []
        viewport.animationToggled.connect(toggles.append)
        viewport._play_button.click()
        assert toggles[-1] is True
        assert viewport._play_button.text() == "■ Stop"
        viewport._play_button.click()
        assert toggles[-1] is False
        assert viewport._play_button.text() == "▶ Play"
    finally:
        viewport._vtk_widget.Finalize()
        viewport.close()
        viewport.deleteLater()
        app.processEvents()
