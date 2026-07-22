from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from urdf_maker.model import JointSpec
from urdf_maker.ui.editors import (
    JointEditorWidget,
    NewLinkDialog,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


class _IgnoredWheelEvent:
    def __init__(self) -> None:
        self.ignored = False

    def ignore(self) -> None:
        self.ignored = True


def test_joint_editor_ignores_wheel_on_value_controls() -> None:
    _app()
    editor = JointEditorWidget()
    controls = (
        editor.type_combo,
        editor.origin_editor.spins[0],
        editor.axis_editor.spins[0],
        editor.lower_spin,
        editor.upper_spin,
        editor.effort_spin,
        editor.velocity_spin,
        editor.damping_spin,
        editor.friction_spin,
        editor.position_spin,
        editor.step_spin,
        editor.position_slider,
        editor.state_spin,
        editor.drive_source_combo,
        editor.drive_max_rpm_spin,
        editor.drive_deadband_spin,
    )

    for control in controls:
        event = _IgnoredWheelEvent()
        control.wheelEvent(event)
        assert event.ignored


def test_continuous_joint_can_be_displayed() -> None:
    _app()
    editor = JointEditorWidget()
    editor.set_link_names(["base", "wheel"])
    joint = JointSpec("wheel_joint", "continuous", "base", "wheel", axis=(0, 0, 1))
    editor.set_joint(joint)
    values = editor.current_values()
    assert values["type"] == "continuous"
    assert math.isclose(values["lower"], -math.pi)
    assert math.isclose(values["upper"], math.pi)


def test_joint_editor_round_trips_simulation_physics_values() -> None:
    _app()
    editor = JointEditorWidget()
    editor.set_link_names(["base", "slider"])
    editor.set_joint(
        JointSpec(
            "slide",
            "prismatic",
            "base",
            "slider",
            lower=0.0,
            upper=0.2,
            effort=750.0,
            velocity=0.15,
            damping=2.5,
            friction=0.4,
        )
    )

    values = editor.current_values()

    assert values["effort"] == 750.0
    assert math.isclose(values["velocity"], 0.15)
    assert values["damping"] == 2.5
    assert values["friction"] == 0.4
    assert editor.velocity_units.text() == "mm/s"


def test_joint_editor_can_flip_axis_direction() -> None:
    _app()
    editor = JointEditorWidget()
    editor.set_link_names(["base", "wheel"])
    editor.set_joint(
        JointSpec("wheel_joint", "revolute", "base", "wheel", axis=(0, 0, 1))
    )
    previews: list[object] = []
    editor.axisPreviewRequested.connect(previews.append)

    editor.flip_axis_button.click()

    np.testing.assert_allclose(editor.axis_editor.value(), (0.0, 0.0, -1.0))
    np.testing.assert_allclose(previews[-1], (0.0, 0.0, -1.0))


def test_joint_editor_configures_automatic_reversed_mimic() -> None:
    _app()
    source = JointSpec(
        "handle_joint",
        "revolute",
        "base",
        "handle",
        lower=-math.pi,
        upper=math.pi,
    )
    target = JointSpec(
        "axle_joint",
        "prismatic",
        "base",
        "axle",
        lower=-0.05,
        upper=0.05,
    )
    editor = JointEditorWidget()
    editor.set_link_names(["base", "handle", "axle"])
    editor.set_joint_specs([source, target])
    editor.set_joint(target)

    editor.mimic_enable.setChecked(True)
    editor.mimic_source_combo.setCurrentText("handle_joint")
    editor.mimic_reverse_check.setChecked(True)
    values = editor.current_values()

    assert values["mimic_joint"] == "handle_joint"
    assert values["mimic_auto"] is True
    assert values["mimic_reverse"] is True
    assert not editor.preview_box.isEnabled()
    assert "-180.000°" in editor.mimic_formula_label.text()
    assert "50.000mm" in editor.mimic_formula_label.text()


def test_floating_joint_disables_scalar_preview() -> None:
    _app()
    editor = JointEditorWidget()
    editor.set_link_names(["world", "object"])
    editor.set_joint(JointSpec("floating", "floating", "world", "object"))
    assert not editor.preview_box.isEnabled()


def test_joint_editor_configures_direction_lever_speed_drive() -> None:
    _app()
    lever = JointSpec(
        "direction_lever",
        "revolute",
        "base",
        "lever",
        lower=math.radians(-20.0),
        upper=math.radians(20.0),
    )
    wheel = JointSpec("wheel_joint", "continuous", "base", "wheel")
    editor = JointEditorWidget()
    editor.set_link_names(["base", "lever", "wheel"])
    editor.set_joint_specs([lever, wheel])
    editor.set_joint(wheel)

    editor.drive_enable.setChecked(True)
    editor.drive_source_combo.setCurrentText("direction_lever")
    editor.drive_max_rpm_spin.setValue(120.0)
    editor.drive_deadband_spin.setValue(5.0)
    editor.drive_reverse_check.setChecked(True)
    values = editor.current_values()

    assert values["drive_source_joint"] == "direction_lever"
    assert math.isclose(values["drive_max_velocity"], 4.0 * math.pi)
    assert math.isclose(values["drive_deadband"], 0.05)
    assert values["drive_reverse"] is True
    assert not editor.preview_box.isEnabled()
    assert "중앙 (0.000°) → 정지" in editor.drive_formula_label.text()


def test_switching_joint_updates_range_before_position() -> None:
    _app()
    editor = JointEditorWidget()
    editor.set_link_names(["base", "first", "second"])
    editor.set_joint(
        JointSpec(
            "small",
            "prismatic",
            "base",
            "first",
            lower=0.0,
            upper=0.001,
            position=0.001,
        )
    )
    editor.set_joint(
        JointSpec(
            "large",
            "prismatic",
            "base",
            "second",
            lower=-0.1,
            upper=0.1,
            position=0.05,
        )
    )
    assert math.isclose(editor.position_spin.value(), 50.0)


def test_prismatic_slider_and_nudge_emit_si_positions() -> None:
    _app()
    editor = JointEditorWidget()
    editor.set_link_names(["base", "slider"])
    editor.set_joint(
        JointSpec(
            "slide",
            "prismatic",
            "base",
            "slider",
            lower=-0.1,
            upper=0.1,
        )
    )
    emitted: list[float] = []
    editor.positionChanged.connect(emitted.append)

    editor.position_slider.setValue(750)
    assert math.isclose(emitted[-1], 0.05)
    editor.step_spin.setValue(5.0)
    editor.plus_button.click()
    assert math.isclose(emitted[-1], 0.055)


def test_normalized_state_maps_zero_and_one_to_actual_joint_values() -> None:
    _app()
    editor = JointEditorWidget()
    editor.set_link_names(["base", "slider"])
    editor.set_joint(
        JointSpec(
            "slide",
            "prismatic",
            "base",
            "slider",
            lower=-0.02,
            upper=0.08,
            position=0.03,
        )
    )
    emitted: list[float] = []
    editor.positionChanged.connect(emitted.append)

    assert math.isclose(editor.state_spin.value(), 0.5)
    editor.state_spin.setValue(0.0)
    assert math.isclose(editor.position_spin.value(), -20.0)
    assert math.isclose(emitted[-1], -0.02)
    editor.state_spin.setValue(1.0)
    assert math.isclose(editor.position_spin.value(), 80.0)
    assert math.isclose(emitted[-1], 0.08)


def test_generic_new_link_dialog_converts_units() -> None:
    _app()
    dialog = NewLinkDialog(
        ["base_link", "mast"],
        default_parent="mast",
        selected_count=3,
    )
    assert dialog.parent_combo.isEnabled()
    dialog.parent_combo.setCurrentText("base_link")
    dialog.name_edit.setText("fork_carriage")
    dialog.type_combo.setCurrentText("revolute")
    dialog.set_candidate_origin((0.4, -0.2, 0.1))
    dialog.select_candidate_axis("Z")
    dialog.lower_spin.setValue(-90.0)
    dialog.upper_spin.setValue(45.0)
    values = dialog.values()

    assert values["link_name"] == "fork_carriage"
    assert values["parent"] == "base_link"
    assert values["joint_type"] == "revolute"
    assert tuple(values["axis"]) == (0.0, 0.0, 1.0)
    np.testing.assert_allclose(values["origin_xyz"], (0.4, -0.2, 0.1))
    assert math.isclose(values["lower"], -math.pi / 2.0)
    assert math.isclose(values["upper"], math.pi / 4.0)


def test_new_link_dialog_recommends_geometry_normal_and_previews_rotation() -> None:
    _app()
    dialog = NewLinkDialog(
        ["base_link"],
        default_parent="base_link",
        selected_count=1,
    )
    assert not dialog.isModal()
    assert dialog.windowModality() == Qt.WindowModality.NonModal
    dialog.set_axis_candidates(
        {
            "A": (1.0, 0.0, 0.0),
            "B": (0.0, 1.0, 0.0),
            "C": (0.0, 0.0, 1.0),
        }
    )
    assert tuple(dialog.axis_editor.value()) == (1.0, 0.0, 0.0)

    dialog.type_combo.setCurrentText("revolute")
    assert tuple(dialog.axis_editor.value()) == (0.0, 0.0, 1.0)
    assert dialog.matching_candidate((0.0, 0.0, -1.0)) == "C"
    assert dialog.geometry_axis_buttons["C"].isChecked()
    assert not dialog.geometry_axis_buttons["A"].isChecked()
    assert dialog.advanced_axis_box.isHidden()
    dialog.flip_axis_direction()
    assert tuple(dialog.axis_editor.value()) == (0.0, 0.0, -1.0)
    assert "반대 방향" in dialog.selected_axis_label.text()
    assert dialog.geometry_axis_buttons["C"].isChecked()
    dialog.lower_spin.setValue(-180.0)
    dialog.upper_spin.setValue(180.0)
    dialog.preview_slider.setValue(100)
    assert math.isclose(dialog.preview_position_si(), math.pi)
    assert dialog.preview_value_label.text() == "180°"
    dialog.preview_slider.setValue(-100)
    assert math.isclose(dialog.preview_position_si(), -math.pi)
    assert dialog.preview_value_label.text() == "-180°"
    dialog.reset_motion_preview()
    assert dialog.preview_slider.value() == 0
    assert math.isclose(dialog.preview_position_si(), 0.0)
    assert dialog.preview_value_label.text() == "0°"

    dialog.set_candidate_origin((0.4, -0.2, 0.1))
    dialog.origin_editor.setValue((410.0, -190.0, 120.0))
    np.testing.assert_allclose(dialog.values()["origin_xyz"], (0.41, -0.19, 0.12))
    dialog.reset_candidate_origin()
    np.testing.assert_allclose(dialog.values()["origin_xyz"], (0.4, -0.2, 0.1))


def test_new_link_dialog_accepts_axis_edits_from_3d_handles() -> None:
    _app()
    dialog = NewLinkDialog(
        ["base_link"],
        default_parent="base_link",
        selected_count=1,
    )
    dialog.set_axis_candidates({"A": (0.0, 0.0, 1.0)})
    previews: list[object] = []
    dialog.axisPreviewRequested.connect(previews.append)

    dialog.axis_edit_button.setChecked(True)
    dialog.set_axis_from_3d((0.12, -0.04, 0.08), (0.0, 3.0, 4.0))

    np.testing.assert_allclose(dialog.values()["origin_xyz"], (0.12, -0.04, 0.08))
    np.testing.assert_allclose(dialog.values()["axis"], (0.0, 0.6, 0.8))
    np.testing.assert_allclose(previews[-1], (0.0, 0.6, 0.8))
    assert "직접 조정 종료" in dialog.axis_edit_button.text()


def test_new_link_dialog_supports_negative_axis_and_validates_zero_one_order() -> None:
    _app()
    dialog = NewLinkDialog(
        ["base_link"],
        default_parent="base_link",
        selected_count=2,
        lock_parent=True,
    )
    assert not dialog.parent_combo.isEnabled()
    dialog.axis_editor.setValue((-1.0, 0.0, 0.0))
    values = dialog.values()
    assert tuple(values["axis"]) == (-1.0, 0.0, 0.0)
    assert math.isclose(values["lower"], 0.0)
    assert math.isclose(values["upper"], 0.1)

    dialog.lower_spin.setValue(120.0)
    dialog.upper_spin.setValue(20.0)
    try:
        dialog.values()
    except ValueError as exc:
        assert "상태 0" in str(exc)
    else:  # pragma: no cover - the dialog must reject reversed URDF limits
        raise AssertionError("reversed state endpoints were accepted")
