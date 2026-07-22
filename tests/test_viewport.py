from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from urdf_maker.model import ScenePart
from urdf_maker.ui.viewport import ViewportWidget


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


def test_replacing_parts_preserves_camera_and_candidate_axes_can_be_highlighted() -> None:
    app = _app()
    viewport = ViewportWidget()
    try:
        viewport.set_parts([_part("one")])
        camera = viewport.renderer.GetActiveCamera()
        camera.SetPosition(7.0, -5.0, 3.0)
        camera.SetFocalPoint(0.4, 0.2, 0.1)
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.SetParallelScale(0.37)
        before = viewport.capture_camera_state()

        viewport.set_parts([_part("two")])
        after = viewport.capture_camera_state()
        np.testing.assert_allclose(after["position"], before["position"])
        np.testing.assert_allclose(after["focal_point"], before["focal_point"])
        assert np.isclose(after["parallel_scale"], before["parallel_scale"])

        viewport.set_candidate_axes(
            (0.5, 0.5, 0.0),
            {"A": (1, 0, 0), "B": (0, 1, 0), "C": (0, 0, 1)},
            0.25,
            selected="B",
        )
        assert set(viewport._candidate_axis_actors) == {"A", "B", "C"}
        assert (
            viewport._candidate_axis_actors["B"].GetProperty().GetLineWidth()
            > viewport._candidate_axis_actors["A"].GetProperty().GetLineWidth()
        )
        viewport.highlight_candidate_axis("C")
        assert (
            viewport._candidate_axis_actors["C"].GetProperty().GetLineWidth()
            > viewport._candidate_axis_actors["B"].GetProperty().GetLineWidth()
        )
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
