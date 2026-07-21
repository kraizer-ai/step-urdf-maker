from __future__ import annotations

from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication, QDialog, QDockWidget, QMessageBox

from urdf_maker.model import JointSpec, LinkSpec, RobotProject, ScenePart
from urdf_maker.ui.editors import NewLinkDialog
from urdf_maker.ui.main_window import MainWindow, ROLE_ID, _stable_part_display_color


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _part(identifier: str, offset: float) -> ScenePart:
    vertices = np.asarray(
        (
            (offset, 0.0, 0.0),
            (offset + 0.01, 0.0, 0.0),
            (offset, 0.01, 0.0),
        ),
        dtype=float,
    )
    return ScenePart(
        identifier,
        identifier,
        vertices,
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )


def test_part_display_color_is_stable_for_the_same_part_id() -> None:
    first = _stable_part_display_color("part_001")
    second = _stable_part_display_color("part_001")
    other = _stable_part_display_color("part_002")

    assert first == second
    assert first != other


def test_maximize_keeps_one_attached_joint_inspector() -> None:
    app = _app()
    window = MainWindow()
    try:
        window.showMaximized()
        app.processEvents()
        window.showNormal()
        app.processEvents()

        inspector_docks = [
            dock
            for dock in window.findChildren(QDockWidget)
            if dock.objectName() == "jointInspectorDock"
        ]
        assert inspector_docks == [window.inspector_dock]
        assert not (
            window.dockOptions() & window.DockOption.AllowTabbedDocks
        )
        assert window.dockWidgetArea(window.inspector_dock) == window.inspector_dock.allowedAreas()
    finally:
        window.viewport._vtk_widget.Finalize()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_close_does_not_show_save_or_discard_prompt() -> None:
    app = _app()
    window = MainWindow()
    try:
        window._dirty = True
        with patch("urdf_maker.ui.main_window.QMessageBox.question") as question:
            assert window.close()
        question.assert_not_called()
    finally:
        window.viewport._vtk_widget.Finalize()
        window.deleteLater()
        app.processEvents()


def test_link_selection_centers_axis_marker_and_demo_restores_position() -> None:
    app = _app()
    base = _part("base_part", 0.0)
    moving = _part("moving_part", 2.0)
    joint = JointSpec(
        "moving_joint",
        "prismatic",
        "base_link",
        "moving_link",
        origin_xyz=(0.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        lower=0.0,
        upper=1.0,
        position=0.25,
    )
    project = RobotProject(
        "demo",
        parts=[base, moving],
        links=[
            LinkSpec("base_link", [base.id]),
            LinkSpec("moving_link", [moving.id]),
        ],
        joints=[joint],
        root_link="base_link",
    )
    window = MainWindow()
    try:
        window._set_project(project, None)
        window._select_link_item("moving_link")
        matrix = window.viewport._axis_actor.GetUserMatrix()
        transformed = project.transformed_part_vertices("moving_part")
        expected_center = (transformed.min(axis=0) + transformed.max(axis=0)) * 0.5
        np.testing.assert_allclose(
            [matrix.GetElement(index, 3) for index in range(3)],
            expected_center,
        )

        original_camera = np.asarray(
            window.viewport.renderer.GetActiveCamera().GetPosition()
        )
        window.viewport._play_button.click()
        assert window._demo_timer.isActive()
        window._demo_started_at -= 1.0
        window._demo_last_tick -= 0.1
        window._advance_demo_animation()
        assert not np.isclose(joint.position, 0.25)
        assert not np.allclose(
            window.viewport.renderer.GetActiveCamera().GetPosition(),
            original_camera,
        )

        window.viewport._play_button.click()
        assert not window._demo_timer.isActive()
        assert np.isclose(joint.position, 0.25)
        assert window.viewport._play_button.text() == "▶ Play"
    finally:
        window.viewport._vtk_widget.Finalize()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_revolute_axis_marker_passes_through_actual_joint_origin() -> None:
    app = _app()
    base = _part("base_part", 0.0)
    moving = _part("moving_part", 2.0)
    origin = np.asarray((0.45, -0.2, 0.3))
    project = RobotProject(
        "rotation_axis",
        parts=[base, moving],
        links=[
            LinkSpec("base_link", [base.id]),
            LinkSpec("moving_link", [moving.id]),
        ],
        joints=[
            JointSpec(
                "moving_joint",
                "revolute",
                "base_link",
                "moving_link",
                origin_xyz=origin,
                axis=(0.0, 0.0, 1.0),
                lower=-1.0,
                upper=1.0,
            )
        ],
        root_link="base_link",
    )
    window = MainWindow()
    try:
        window._set_project(project, None)
        window._select_link_item("moving_link")

        matrix = window.viewport._axis_actor.GetUserMatrix()
        np.testing.assert_allclose(
            [matrix.GetElement(index, 3) for index in range(3)],
            origin,
        )
        assert window.viewport._axis_bidirectional is True
    finally:
        window.viewport._vtk_widget.Finalize()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_new_child_preview_uses_selected_bundle_bbox_center_and_parent_context() -> None:
    app = _app()
    base = _part("base_part", 0.0)
    handle = _part("handle_part", 2.0)
    project = RobotProject(
        "candidate_axes",
        parts=[base, handle],
        links=[LinkSpec("base_link", [base.id])],
        root_link="base_link",
    )
    window = MainWindow()
    dialog = None
    try:
        window._set_project(project, None)
        dialog = NewLinkDialog(
            project.links.keys(),
            default_parent="base_link",
            selected_count=1,
            parent=window,
        )
        window._preview_new_link_candidates(
            dialog,
            "base_link",
            ["handle_part"],
        )

        expected_center = (handle.vertices_zero.min(axis=0) + handle.vertices_zero.max(axis=0)) * 0.5
        np.testing.assert_allclose(dialog._candidate_origin, expected_center)
        assert set(window.viewport._candidate_axis_actors) == {"X", "Y", "Z"}
        assert window.viewport._parts["base_part"].actor.GetVisibility()
        assert window.viewport._parts["handle_part"].actor.GetVisibility()
        assert window.viewport.selected_ids() == ["handle_part"]

        assert window._create_moving_link(
            "handle_link",
            ["handle_part"],
            parent="base_link",
            joint_name="handle_joint",
            axis=(0.0, 0.0, 1.0),
            origin_xyz=dialog._candidate_origin,
            joint_type="revolute",
            lower=-1.0,
            upper=1.0,
        )
        np.testing.assert_allclose(
            project.joint("handle_joint").origin_xyz,
            expected_center,
        )
    finally:
        if dialog is not None:
            dialog.close()
            dialog.deleteLater()
        window.viewport._vtk_widget.Finalize()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_new_manual_tree_and_nested_children_follow_selected_parent() -> None:
    _app()
    parts = [
        _part("base_part", 0.0),
        _part("base_extra_1", 0.03),
        _part("base_extra_2", 0.06),
        _part("base_extra_3", 0.09),
        _part("hand_part", 0.12),
        _part("finger_part", 0.2),
    ]
    project = RobotProject(
        "picker",
        parts=parts,
        links=[LinkSpec("base_link", [part.id for part in parts])],
        root_link="base_link",
    )
    window = MainWindow()
    try:
        window._set_project(project, None)
        window._set_selected_parts(["base_part"])
        with (
            patch(
                "urdf_maker.ui.main_window.QInputDialog.getText",
                return_value=("manual_base", True),
            ),
            patch(
                "urdf_maker.ui.main_window.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ),
        ):
            window.new_manual_tree()

        assert list(project.links) == ["manual_base"]
        assert project.root_link == "manual_base"
        assert project.parts["base_part"].link_name is None
        assert project.parts["hand_part"].link_name is None
        assert window._selected_part_ids() == []

        base_selection = [
            "base_part",
            "base_extra_1",
            "base_extra_2",
            "base_extra_3",
        ]
        window._set_selected_parts(base_selection)
        window.assign_selection_to_current_link()
        assert set(project.links["manual_base"].part_ids) == set(base_selection)
        assert project.parts["hand_part"].link_name is None
        assert project.parts["finger_part"].link_name is None
        assert project.parts["base_part"].visible
        assert "base_part" not in window.viewport._parts
        listed_ids = {
            str(item.data(0, ROLE_ID))
            for item in window._walk_part_tree()
            if item.data(0, ROLE_ID) is not None
        }
        assert listed_ids == {"hand_part", "finger_part"}

        window.hide_assigned_action.setChecked(False)
        assert not window.assigned_visibility_check.isChecked()
        assert "base_part" in window.viewport._parts
        camera = window.viewport.renderer.GetActiveCamera()
        camera.SetPosition(5.0, -4.0, 2.5)
        camera.SetFocalPoint(0.08, 0.03, 0.01)
        camera.SetParallelScale(0.42)
        camera_before_visibility = window.viewport.capture_camera_state()
        window.assigned_visibility_check.setChecked(True)
        assert window.hide_assigned_action.isChecked()
        assert "base_part" not in window.viewport._parts
        camera_after_visibility = window.viewport.capture_camera_state()
        np.testing.assert_allclose(
            camera_after_visibility["position"],
            camera_before_visibility["position"],
        )
        np.testing.assert_allclose(
            camera_after_visibility["focal_point"],
            camera_before_visibility["focal_point"],
        )
        assert np.isclose(
            camera_after_visibility["parallel_scale"],
            camera_before_visibility["parallel_scale"],
        )

        assert window._create_moving_link(
            "left_hand",
            ["hand_part"],
            parent="manual_base",
            joint_name="left_hand_joint",
            axis=(-1.0, 0.0, 0.0),
            joint_type="prismatic",
            lower=0.0,
            upper=0.12,
        )
        assert window._create_moving_link(
            "left_finger",
            ["finger_part"],
            parent="left_hand",
            joint_name="left_finger_joint",
            axis=(0.0, 0.0, -1.0),
            joint_type="prismatic",
            lower=0.0,
            upper=0.08,
        )

        assert "left_finger (1개)" in window.link_parts_label.text()
        assert window.link_parts_list.count() == 1
        member_item = window.link_parts_list.item(0)
        assert member_item.data(ROLE_ID) == "finger_part"
        member_item.setSelected(True)
        window.unassign_selection_from_current_link()
        assert project.parts["finger_part"].link_name is None
        assert project.links["left_finger"].part_ids == []
        assert project.joint_for_child("left_finger") is not None
        assert "finger_part" in window.viewport._parts
        assert window._selected_part_ids() == ["finger_part"]

        window.assign_selection_to_current_link()
        assert project.parts["finger_part"].link_name == "left_finger"
        assert project.links["left_finger"].part_ids == ["finger_part"]
        assert window.link_parts_list.count() == 1

        assert project.joint_for_child("left_hand").parent == "manual_base"
        assert project.joint_for_child("left_finger").parent == "left_hand"
        assert project.validate(check_names=False) == []
        root_item = window.link_tree.topLevelItem(0)
        assert root_item.data(0, ROLE_ID) == "manual_base"
        assert root_item.child(0).data(0, ROLE_ID) == "left_hand"
        assert root_item.child(0).child(0).data(0, ROLE_ID) == "left_finger"
        assert "직선 −X" in root_item.child(0).text(1)
        assert "0:0 → 1:120 mm" in root_item.child(0).text(1)

        window.current_link = "left_hand"
        camera.SetPosition(3.2, -2.1, 1.7)
        camera.SetFocalPoint(0.1, 0.05, 0.0)
        camera_before_dialog = window.viewport.capture_camera_state()
        with patch("urdf_maker.ui.main_window.NewLinkDialog") as dialog_class:
            dialog = dialog_class.return_value
            dialog.exec.return_value = QDialog.DialogCode.Accepted
            dialog.values.return_value = {
                "link_name": "right_hand",
                "parent": "manual_base",
                "joint_type": "prismatic",
                "axis": (1.0, 0.0, 0.0),
                "lower": 0.0,
                "upper": 0.12,
            }
            window.add_child_link_from_selection()

        camera_after_dialog = window.viewport.capture_camera_state()
        np.testing.assert_allclose(
            camera_after_dialog["position"],
            camera_before_dialog["position"],
        )
        np.testing.assert_allclose(
            camera_after_dialog["focal_point"],
            camera_before_dialog["focal_point"],
        )

        assert dialog_class.call_args.kwargs["default_parent"] == "left_hand"
        assert dialog_class.call_args.kwargs["lock_parent"] is False
        assert project.joint_for_child("right_hand").parent == "manual_base"

        window.current_link = "left_hand"
        with patch(
            "urdf_maker.ui.main_window.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            window.delete_current_link()
        assert "left_hand" not in project.links
        assert project.parts["hand_part"].link_name is None
        assert "hand_part" not in project.links["manual_base"].part_ids
        assert window._selected_part_ids() == ["hand_part"]
        assert project.joint_for_child("left_finger").parent == "manual_base"
        assert project.validate(check_names=False) == []

        assert window._create_moving_link(
            "empty_child",
            [],
            parent="manual_base",
            joint_name="empty_child_joint",
            axis=(1.0, 0.0, 0.0),
            joint_type="prismatic",
            lower=0.0,
            upper=0.05,
            allow_empty=True,
        )
        assert project.links["empty_child"].part_ids == []
        assert project.validate(check_names=False) == []
    finally:
        window._dirty = False
        window.viewport._vtk_widget.Finalize()
        window.close()
        window.deleteLater()
        _app().processEvents()
