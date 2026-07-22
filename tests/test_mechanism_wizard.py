from __future__ import annotations

import math

from PySide6.QtWidgets import QApplication

from urdf_maker.model import JointSpec
from urdf_maker.ui.editors import NewLinkDialog
from urdf_maker.ui.wizards import MechanismWizard, SimulationExportDialog


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_hinge_preset_hands_editable_defaults_to_3d_dialog() -> None:
    _app()
    wizard = MechanismWizard(
        ["base_link"],
        [],
        default_parent="base_link",
        selected_count=2,
    )
    values = wizard.values()

    assert values["preset"] == "hinge"
    assert values["joint_type"] == "revolute"
    assert math.isclose(values["lower_si"], 0.0)
    assert math.isclose(values["upper_si"], math.pi / 2.0)
    assert values["state_0"] == "닫힘"
    assert values["state_1"] == "열림"

    dialog = NewLinkDialog(
        ["base_link"],
        default_parent="base_link",
        selected_count=2,
    )
    dialog.apply_mechanism_preset(values)
    try:
        assert dialog.type_combo.currentText() == "revolute"
        assert dialog.name_edit.text() == "hinged_link"
        assert dialog.lower_spin.value() == 0.0
        assert dialog.upper_spin.value() == 90.0
        assert dialog.lower_state_label.text() == "닫힘"
        assert dialog.upper_state_label.text() == "열림"
        assert dialog.preset_hint.isVisibleTo(dialog)
    finally:
        dialog.deleteLater()


def test_coupled_preset_selects_standard_mimic_source() -> None:
    _app()
    source = JointSpec(
        "driver_joint",
        "revolute",
        "base_link",
        "driver_link",
        lower=-1.0,
        upper=1.0,
    )
    wizard = MechanismWizard(
        ["base_link", "driver_link"],
        [source],
        default_parent="base_link",
        selected_count=1,
    )
    wizard.kind_combo.setCurrentIndex(wizard.kind_combo.findData("coupled"))
    wizard.source_combo.setCurrentText("driver_joint")
    wizard.reverse_check.setChecked(True)

    values = wizard.values()

    assert values["source_mode"] == "mimic"
    assert values["source_joint"] == "driver_joint"
    assert values["reverse"] is True
    assert values["simulation_role"] == "mimic"


def test_conveyor_preset_can_remain_an_independent_continuous_joint() -> None:
    _app()
    wizard = MechanismWizard(
        ["base_link"],
        [],
        default_parent="base_link",
        selected_count=1,
    )
    wizard.kind_combo.setCurrentIndex(wizard.kind_combo.findData("conveyor"))
    values = wizard.values()

    assert values["joint_type"] == "continuous"
    assert values["source_joint"] is None
    assert values["simulation_role"] == "conveyor_velocity"
    assert math.isclose(values["max_rpm"], 60.0)


def test_simulation_export_requires_explicit_approximate_inertia_choice() -> None:
    _app()
    dialog = SimulationExportDialog()
    assert dialog.values()["include_inertial"] is False
    assert not dialog.density_spin.isEnabled()

    dialog.inertial_check.setChecked(True)
    dialog.density_spin.setValue(7800.0)

    assert dialog.values()["include_inertial"] is True
    assert dialog.values()["density"] == 7800.0
    assert dialog.density_spin.isEnabled()
