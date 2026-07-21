from __future__ import annotations

import math

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
        editor.position_spin,
        editor.step_spin,
        editor.position_slider,
        editor.state_spin,
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


def test_floating_joint_disables_scalar_preview() -> None:
    _app()
    editor = JointEditorWidget()
    editor.set_link_names(["world", "object"])
    editor.set_joint(JointSpec("floating", "floating", "world", "object"))
    assert not editor.preview_box.isEnabled()


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
    dialog.lower_spin.setValue(-90.0)
    dialog.upper_spin.setValue(45.0)
    values = dialog.values()

    assert values["link_name"] == "fork_carriage"
    assert values["parent"] == "base_link"
    assert values["joint_type"] == "revolute"
    assert math.isclose(values["lower"], -math.pi / 2.0)
    assert math.isclose(values["upper"], math.pi / 4.0)


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
