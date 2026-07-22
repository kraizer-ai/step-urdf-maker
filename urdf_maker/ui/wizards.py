from __future__ import annotations

from typing import Any, Iterable

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)


MECHANISM_PRESETS: dict[str, dict[str, Any]] = {
    "hinge": {
        "title": "문·뚜껑·레버 (회전)",
        "description": (
            "힌지 중심을 기준으로 정해진 각도만큼 움직입니다. "
            "박스 뚜껑, 도어, 손잡이와 제한 회전축에 사용합니다."
        ),
        "joint_type": "revolute",
        "link_name": "hinged_link",
        "lower": 0.0,
        "upper": 90.0,
        "state_0": "닫힘",
        "state_1": "열림",
        "source_mode": "none",
        "effort": 100.0,
        "velocity": 45.0,
        "damping": 0.1,
        "friction": 0.0,
        "simulation_role": "position",
    },
    "slider": {
        "title": "슬라이더·리프트·서랍 (직선)",
        "description": (
            "한 축을 따라 제한 거리만큼 움직입니다. 리니어 레일, 실린더, "
            "서랍, 승강 장치와 포크 캐리지에 사용합니다."
        ),
        "joint_type": "prismatic",
        "link_name": "slider_link",
        "lower": 0.0,
        "upper": 100.0,
        "state_0": "수축",
        "state_1": "확장",
        "source_mode": "none",
        "effort": 1000.0,
        "velocity": 100.0,
        "damping": 1.0,
        "friction": 0.0,
        "simulation_role": "position",
    },
    "rotary": {
        "title": "바퀴·롤러·턴테이블 (연속 회전)",
        "description": (
            "회전 한계가 없는 축을 만듭니다. 바퀴, 팬, 롤러, 스핀들과 "
            "턴테이블에 사용하며 실제 구동기는 시뮬레이터에서 연결합니다."
        ),
        "joint_type": "continuous",
        "link_name": "rotary_link",
        "lower": -180.0,
        "upper": 180.0,
        "state_0": "역회전 미리보기",
        "state_1": "정회전 미리보기",
        "source_mode": "none",
        "effort": 100.0,
        "velocity": 360.0,
        "damping": 0.01,
        "friction": 0.0,
        "simulation_role": "velocity",
    },
    "coupled": {
        "title": "그리퍼·대칭 부품 (mimic 연동)",
        "description": (
            "이미 만든 구동 관절의 위치를 따라 움직이는 자식 관절을 만듭니다. "
            "2핑거 그리퍼, 양문과 선형 비율로 동기화되는 축에 사용합니다."
        ),
        "joint_type": "prismatic",
        "link_name": "coupled_link",
        "lower": 0.0,
        "upper": 50.0,
        "state_0": "닫힘",
        "state_1": "열림",
        "source_mode": "mimic",
        "effort": 100.0,
        "velocity": 100.0,
        "damping": 0.2,
        "friction": 0.0,
        "simulation_role": "mimic",
    },
    "conveyor": {
        "title": "컨베이어 롤러·구동축",
        "description": (
            "벨트 또는 롤러의 연속 회전축을 만듭니다. 입력 관절을 선택하면 "
            "앱의 Play 미리보기에서 입력 위치에 비례한 속도로 회전합니다."
        ),
        "joint_type": "continuous",
        "link_name": "conveyor_roller_link",
        "lower": -180.0,
        "upper": 180.0,
        "state_0": "역방향 미리보기",
        "state_1": "정방향 미리보기",
        "source_mode": "drive",
        "effort": 200.0,
        "velocity": 180.0,
        "damping": 0.02,
        "friction": 0.0,
        "simulation_role": "conveyor_velocity",
    },
}


class MechanismWizard(QWizard):
    """Choose a common mechanism, then hand off to precise 3D joint authoring."""

    def __init__(
        self,
        link_names: Iterable[str],
        joint_specs: Iterable[Any],
        *,
        default_parent: str | None,
        selected_count: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._link_names = [str(name) for name in link_names]
        self._joint_specs = {
            str(getattr(joint, "name", "")): joint
            for joint in joint_specs
            if str(getattr(joint, "name", ""))
        }
        self.setWindowTitle("대표 기구 마법사")
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setMinimumWidth(560)
        self.setOption(QWizard.WizardOption.NoBackButtonOnStartPage, True)
        self.setButtonText(QWizard.WizardButton.FinishButton, "3D 축 정밀 설정으로 이동")

        choose_page = QWizardPage()
        choose_page.setTitle("1. 만들 기구 선택")
        choose_layout = QVBoxLayout(choose_page)
        summary = QLabel(
            f"선택한 형상 {selected_count}개를 하나의 움직이는 링크로 만듭니다. "
            "마법사는 안전한 시작값을 채우며, 다음 3D 단계에서 축·중심·범위를 다시 조정할 수 있습니다."
        )
        summary.setWordWrap(True)
        choose_layout.addWidget(summary)
        self.kind_combo = QComboBox()
        for key, preset in MECHANISM_PRESETS.items():
            self.kind_combo.addItem(str(preset["title"]), key)
        choose_layout.addWidget(self.kind_combo)
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(
            "QLabel { background: #eef5fb; border: 1px solid #c9d9e6; "
            "border-radius: 5px; padding: 10px; }"
        )
        choose_layout.addWidget(self.description_label)
        choose_layout.addStretch(1)
        self.addPage(choose_page)

        setup_page = QWizardPage()
        setup_page.setTitle("2. 동작과 시뮬레이션 기본값")
        setup_layout = QVBoxLayout(setup_page)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.parent_combo = QComboBox()
        self.parent_combo.addItems(self._link_names)
        if default_parent in self._link_names:
            self.parent_combo.setCurrentText(str(default_parent))
        self.link_name_edit = QLineEdit()
        self.joint_type_combo = QComboBox()
        self.joint_type_combo.addItems(["prismatic", "revolute", "continuous"])
        form.addRow("부모 링크", self.parent_combo)
        form.addRow("새 링크 이름", self.link_name_edit)
        form.addRow("동작 종류", self.joint_type_combo)

        limits = QGroupBox("상태 0 / 상태 1")
        limits_form = QFormLayout(limits)
        self.state_0_edit = QLineEdit()
        self.state_1_edit = QLineEdit()
        self.lower_spin = QDoubleSpinBox()
        self.upper_spin = QDoubleSpinBox()
        for spin in (self.lower_spin, self.upper_spin):
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(3)
        lower_row = QHBoxLayout()
        lower_row.addWidget(self.lower_spin, 1)
        self.lower_units = QLabel("mm")
        lower_row.addWidget(self.lower_units)
        upper_row = QHBoxLayout()
        upper_row.addWidget(self.upper_spin, 1)
        self.upper_units = QLabel("mm")
        upper_row.addWidget(self.upper_units)
        limits_form.addRow("상태 0 이름", self.state_0_edit)
        limits_form.addRow("상태 0 실제값", lower_row)
        limits_form.addRow("상태 1 이름", self.state_1_edit)
        limits_form.addRow("상태 1 실제값", upper_row)
        form.addRow(limits)
        setup_layout.addLayout(form)

        self.source_box = QGroupBox("연동 입력")
        source_form = QFormLayout(self.source_box)
        self.source_label = QLabel("구동 관절")
        self.source_combo = QComboBox()
        self.reverse_check = QCheckBox("반대 방향으로 연결")
        source_form.addRow(self.source_label, self.source_combo)
        source_form.addRow("방향", self.reverse_check)
        setup_layout.addWidget(self.source_box)

        self.drive_box = QGroupBox("속도 미리보기")
        drive_form = QFormLayout(self.drive_box)
        self.max_rpm_spin = QDoubleSpinBox()
        self.max_rpm_spin.setRange(0.1, 100_000.0)
        self.max_rpm_spin.setDecimals(1)
        self.max_rpm_spin.setValue(60.0)
        self.deadband_spin = QDoubleSpinBox()
        self.deadband_spin.setRange(0.0, 49.0)
        self.deadband_spin.setDecimals(1)
        self.deadband_spin.setSuffix(" %")
        self.deadband_spin.setValue(3.0)
        drive_form.addRow("최대 속도", self.max_rpm_spin)
        drive_form.addRow("입력 중앙 정지 범위", self.deadband_spin)
        setup_layout.addWidget(self.drive_box)

        self.simulation_hint = QLabel()
        self.simulation_hint.setWordWrap(True)
        self.simulation_hint.setStyleSheet(
            "QLabel { background: #fff8df; border: 1px solid #e6d39b; "
            "border-radius: 5px; padding: 8px; }"
        )
        setup_layout.addWidget(self.simulation_hint)
        setup_layout.addStretch(1)
        self.addPage(setup_page)
        self._setup_page = setup_page

        self.kind_combo.currentIndexChanged.connect(self._refresh_preset)
        self.joint_type_combo.currentTextChanged.connect(self._refresh_units)
        self._refresh_preset()

    def _preset(self) -> dict[str, Any]:
        return MECHANISM_PRESETS[str(self.kind_combo.currentData())]

    def _eligible_sources(self, mode: str) -> list[str]:
        allowed = (
            {"revolute", "prismatic"}
            if mode == "drive"
            else {"revolute", "prismatic", "continuous"}
        )
        return [
            name
            for name, joint in self._joint_specs.items()
            if str(getattr(joint, "type", "")) in allowed
        ]

    def _refresh_preset(self) -> None:
        preset = self._preset()
        self.description_label.setText(str(preset["description"]))
        self.link_name_edit.setText(str(preset["link_name"]))
        self.joint_type_combo.setCurrentText(str(preset["joint_type"]))
        self.lower_spin.setValue(float(preset["lower"]))
        self.upper_spin.setValue(float(preset["upper"]))
        self.state_0_edit.setText(str(preset["state_0"]))
        self.state_1_edit.setText(str(preset["state_1"]))
        self._refresh_units()

        mode = str(preset["source_mode"])
        self.source_box.setVisible(mode in {"mimic", "drive"})
        self.drive_box.setVisible(mode == "drive")
        self.source_combo.clear()
        if mode == "drive":
            self.source_label.setText("속도 입력 관절 (선택)")
            self.source_combo.addItem("독립 회전 · 시뮬레이터에서 구동", None)
        elif mode == "mimic":
            self.source_label.setText("구동 관절")
            self.source_combo.addItem("구동 관절 선택", None)
        for name in self._eligible_sources(mode):
            self.source_combo.addItem(name, name)
        self.reverse_check.setText(
            "회전 방향 반전" if mode == "drive" else "반대 방향으로 연동"
        )
        self.reverse_check.setChecked(False)
        self.max_rpm_spin.setValue(60.0)
        role = str(preset["simulation_role"])
        if role == "mimic":
            hint = (
                "표준 URDF <mimic>으로 내보냅니다. 정확한 선형 연동에 적합하며, "
                "폐루프·비선형 기구에는 별도 솔버가 필요합니다."
            )
        elif role == "conveyor_velocity":
            hint = (
                "URDF에는 continuous 롤러 관절을 만듭니다. 벨트 표면이 물체를 운반하는 "
                "물리는 Gazebo 등에서 컨베이어 플러그인 또는 접촉 모델을 추가해야 합니다."
            )
        elif role == "velocity":
            hint = (
                "URDF에는 continuous 관절과 물리 기본값을 만듭니다. 실제 회전속도 명령은 "
                "ros2_control 또는 시뮬레이터 컨트롤러에서 연결합니다."
            )
        else:
            hint = (
                "관절 한계, 속도, 힘, 감쇠와 마찰을 표준 URDF로 내보냅니다. "
                "값은 다음 3D 단계와 일반 관절 편집기에서 다시 수정할 수 있습니다."
            )
        self.simulation_hint.setText(hint)

    def _refresh_units(self) -> None:
        units = "mm" if self.joint_type_combo.currentText() == "prismatic" else "deg"
        self.lower_units.setText(units)
        self.upper_units.setText(units)

    def validateCurrentPage(self) -> bool:  # noqa: N802 - Qt API
        if self.currentPage() is not self._setup_page:
            return True
        name = self.link_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "기구 설정", "새 링크 이름을 입력하세요.")
            return False
        if name in self._link_names:
            QMessageBox.warning(
                self,
                "기구 설정",
                "같은 이름의 링크가 이미 있습니다. 새 링크 이름을 사용하세요.",
            )
            return False
        kind = self.joint_type_combo.currentText()
        if kind != "continuous" and self.lower_spin.value() > self.upper_spin.value():
            QMessageBox.warning(
                self,
                "기구 설정",
                "상태 0 실제값은 상태 1 실제값보다 클 수 없습니다. 반대 방향은 축 반전을 사용하세요.",
            )
            return False
        if self._preset()["source_mode"] == "mimic" and not self.source_combo.currentData():
            QMessageBox.warning(self, "기구 설정", "연동할 구동 관절을 선택하세요.")
            return False
        return True

    def values(self) -> dict[str, Any]:
        preset = self._preset()
        joint_type = self.joint_type_combo.currentText()
        scale = 1000.0 if joint_type == "prismatic" else 180.0 / 3.141592653589793
        source_mode = str(preset["source_mode"])
        velocity_scale = scale
        return {
            "preset": str(self.kind_combo.currentData()),
            "title": str(preset["title"]),
            "parent": self.parent_combo.currentText(),
            "link_name": self.link_name_edit.text().strip(),
            "joint_type": joint_type,
            "lower": self.lower_spin.value(),
            "upper": self.upper_spin.value(),
            "lower_si": self.lower_spin.value() / scale,
            "upper_si": self.upper_spin.value() / scale,
            "state_0": self.state_0_edit.text().strip() or "상태 0",
            "state_1": self.state_1_edit.text().strip() or "상태 1",
            "source_mode": source_mode,
            "source_joint": self.source_combo.currentData()
            if source_mode in {"mimic", "drive"}
            else None,
            "reverse": bool(self.reverse_check.isChecked()),
            "max_rpm": float(self.max_rpm_spin.value()),
            "deadband": float(self.deadband_spin.value()) / 100.0,
            "effort": float(preset["effort"]),
            "velocity": float(preset["velocity"]) / velocity_scale,
            "damping": float(preset["damping"]),
            "friction": float(preset["friction"]),
            "simulation_role": str(preset["simulation_role"]),
        }


class SimulationExportDialog(QDialog):
    """Choose whether to add an explicit approximate dynamics starting point."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("URDF 내보내기 용도")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "표시·기구학 확인용 URDF는 관성 없이도 사용할 수 있지만, Gazebo 같은 "
            "동역학 시뮬레이터의 움직이는 링크에는 질량과 관성이 필요합니다."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.inertial_check = QCheckBox("BBox 기반 근사 질량·관성 생성")
        self.inertial_check.setChecked(False)
        self.inertial_check.setToolTip(
            "각 링크 메시의 경계 상자를 균일한 직육면체로 가정합니다."
        )
        layout.addWidget(self.inertial_check)
        density_row = QHBoxLayout()
        density_row.addWidget(QLabel("가정 밀도"))
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.001, 100_000.0)
        self.density_spin.setDecimals(3)
        self.density_spin.setValue(500.0)
        self.density_spin.setSuffix(" kg/m³")
        density_row.addWidget(self.density_spin, 1)
        layout.addLayout(density_row)
        warning = QLabel(
            "근사 관성은 시뮬레이터에서 모델을 움직여 보기 위한 시작값입니다. "
            "실제 하중·안정성·제어 검증에는 CAD 질량 특성 또는 측정값으로 교체해야 합니다. "
            "collision은 현재 visual과 같은 정밀 메시로 생성됩니다."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            "QLabel { background: #fff8df; border: 1px solid #e6d39b; "
            "border-radius: 5px; padding: 8px; }"
        )
        layout.addWidget(warning)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.inertial_check.toggled.connect(self.density_spin.setEnabled)
        self.density_spin.setEnabled(False)

    def values(self) -> dict[str, Any]:
        return {
            "include_collision": True,
            "include_inertial": bool(self.inertial_check.isChecked()),
            "density": float(self.density_spin.value()),
        }


__all__ = [
    "MECHANISM_PRESETS",
    "MechanismWizard",
    "SimulationExportDialog",
]
