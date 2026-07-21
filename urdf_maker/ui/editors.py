from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterable

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


@contextmanager
def _blocked(*widgets: QWidget):
    previous = [widget.blockSignals(True) for widget in widgets]
    try:
        yield
    finally:
        for widget, state in zip(widgets, previous, strict=True):
            widget.blockSignals(state)


def _make_horizontally_compact(widget: QWidget, minimum_width: int = 64) -> None:
    """Let a value editor shrink without losing keyboard editability.

    Qt derives a QDoubleSpinBox size hint from its complete numeric range.  A
    range such as +/-1,000,000 with decimals can otherwise force a single spin
    box to be about 180 px wide and make the entire inspector overflow.  The
    line edit still scrolls its text while editing at compact widths.
    """

    widget.setMinimumWidth(minimum_width)
    widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)


class _NoWheelComboBox(QComboBox):
    """Keep panel scrolling from accidentally changing a value under the cursor."""

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        event.ignore()


class _NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Numeric editor that deliberately leaves wheel input to its scroll area."""

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        event.ignore()


class _NoWheelSlider(QSlider):
    """Slider changed by drag/keyboard only, never by incidental panel scrolling."""

    def wheelEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        event.ignore()


class NewLinkDialog(QDialog):
    """Create one child link and define its incoming joint motion."""

    parentPreviewRequested = Signal(str)
    axisPreviewRequested = Signal(object)

    def __init__(
        self,
        link_names: Iterable[str],
        *,
        default_parent: str | None,
        selected_count: int,
        lock_parent: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._candidate_origin: np.ndarray | None = None
        self.setWindowTitle("자식 링크와 동작 만들기")
        self.setMinimumWidth(380)
        root = QVBoxLayout(self)
        if selected_count:
            summary_text = (
                f"선택한 형상 {selected_count}개를 하나의 강체 링크로 만들고, "
                "부모에 대한 움직임을 정의합니다."
            )
        else:
            summary_text = (
                "빈 자식 링크를 먼저 만듭니다. 생성 후 형상을 선택하여 "
                "'선택 형상을 현재 링크에 넣기'로 배정할 수 있습니다."
            )
        summary = QLabel(summary_text)
        summary.setWordWrap(True)
        root.addWidget(summary)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.name_edit = QLineEdit("moving_link")
        self.parent_combo = _NoWheelComboBox()
        self.parent_combo.addItems(list(link_names))
        if default_parent:
            self.parent_combo.setCurrentText(default_parent)
        self.parent_combo.setEnabled(not lock_parent)
        if lock_parent:
            self.parent_combo.setToolTip("트리에서 선택한 링크가 부모로 고정됩니다.")
        self.type_combo = _NoWheelComboBox()
        self.type_combo.addItems(["fixed", "prismatic", "revolute", "continuous"])
        self.type_combo.setCurrentText("prismatic")
        self.axis_editor = Vector3Editor(minimum=-1.0, maximum=1.0, decimals=4, step=0.1)
        self.axis_editor.setValue((1.0, 0.0, 0.0))

        axis_box = QWidget()
        axis_layout = QVBoxLayout(axis_box)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        axis_layout.addWidget(self.axis_editor)
        shortcuts = QGridLayout()
        axis_colors = {
            "X": "#dc4943",
            "Y": "#38b95a",
            "Z": "#4b7fe8",
        }
        for index, (label, vector) in enumerate(
            (
                ("+X", (1, 0, 0)),
                ("−X", (-1, 0, 0)),
                ("+Y", (0, 1, 0)),
                ("−Y", (0, -1, 0)),
                ("+Z", (0, 0, 1)),
                ("−Z", (0, 0, -1)),
            )
        ):
            button = QPushButton(label)
            button.setStyleSheet(
                f"QPushButton {{ border: 2px solid {axis_colors[label[-1]]}; }}"
            )
            button.clicked.connect(
                lambda _checked=False, value=vector: self._choose_candidate_axis(value)
            )
            shortcuts.addWidget(button, index % 2, index // 2)
        axis_layout.addLayout(shortcuts)
        axis_hint = QLabel(
            "3D의 빨강 X · 초록 Y · 파랑 Z 후보축과 같은 방향입니다. "
            "축의 반대 회전은 − 버튼을 선택하세요."
        )
        axis_hint.setWordWrap(True)
        axis_layout.addWidget(axis_hint)

        limit_box = QWidget()
        limit_layout = QHBoxLayout(limit_box)
        limit_layout.setContentsMargins(0, 0, 0, 0)
        self.lower_spin = _NoWheelDoubleSpinBox()
        self.upper_spin = _NoWheelDoubleSpinBox()
        for spin in (self.lower_spin, self.upper_spin):
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(3)
        self.lower_spin.setValue(0.0)
        self.upper_spin.setValue(100.0)
        self.units_label = QLabel("mm")
        limit_layout.addWidget(QLabel("상태 0"))
        limit_layout.addWidget(self.lower_spin)
        limit_layout.addWidget(QLabel("상태 1"))
        limit_layout.addWidget(self.upper_spin)
        limit_layout.addWidget(self.units_label)

        form.addRow("링크 이름", self.name_edit)
        form.addRow("부모 링크", self.parent_combo)
        form.addRow("동작 종류", self.type_combo)
        form.addRow("이동/회전 방향", axis_box)
        form.addRow("0/1 실제값", limit_box)
        root.addLayout(form)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)
        self.type_combo.currentTextChanged.connect(self._refresh_type)
        self.parent_combo.currentTextChanged.connect(self.parentPreviewRequested.emit)
        self.axis_editor.valueChanged.connect(self.axisPreviewRequested.emit)
        self._refresh_type(self.type_combo.currentText())

    def set_candidate_origin(self, value: Iterable[float]) -> None:
        origin = np.asarray(tuple(value), dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("Candidate origin must contain three finite values")
        self._candidate_origin = origin

    def select_candidate_axis(self, name: str) -> None:
        vectors = {
            "X": (1.0, 0.0, 0.0),
            "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
        }
        axis = str(name).upper().lstrip("+-")
        if axis in vectors:
            self._choose_candidate_axis(vectors[axis])

    def _choose_candidate_axis(self, value: Iterable[float]) -> None:
        self.axis_editor.setValue(value)
        self.axisPreviewRequested.emit(self.axis_editor.value())

    def _refresh_type(self, joint_type: str) -> None:
        scalar = joint_type in {"prismatic", "revolute", "continuous"}
        self.axis_editor.setEnabled(scalar)
        self.lower_spin.setEnabled(joint_type != "fixed")
        self.upper_spin.setEnabled(joint_type != "fixed")
        angular = joint_type in {"revolute", "continuous"}
        self.units_label.setText("deg" if angular else ("mm" if joint_type == "prismatic" else "—"))
        if joint_type == "continuous":
            self.lower_spin.setValue(-180.0)
            self.upper_spin.setValue(180.0)

    def values(self) -> dict[str, Any]:
        kind = self.type_combo.currentText()
        scale = 1000.0 if kind == "prismatic" else 180.0 / math.pi
        axis = self.axis_editor.value()
        norm = float(np.linalg.norm(axis))
        if kind in {"prismatic", "revolute", "continuous"} and norm < 1.0e-9:
            raise ValueError("관절축은 0 벡터일 수 없습니다.")
        if norm >= 1.0e-9:
            axis /= norm
        lower = self.lower_spin.value() / scale if kind != "fixed" else 0.0
        upper = self.upper_spin.value() / scale if kind != "fixed" else 0.0
        if kind in {"prismatic", "revolute"} and lower > upper:
            raise ValueError("상태 0 실제값은 상태 1 실제값보다 클 수 없습니다. 반대 방향은 −축을 사용하세요.")
        return {
            "link_name": self.name_edit.text().strip(),
            "parent": self.parent_combo.currentText(),
            "joint_type": kind,
            "axis": axis,
            "origin_xyz": (
                None
                if self._candidate_origin is None
                else self._candidate_origin.copy()
            ),
            "lower": lower,
            "upper": upper,
        }


class Vector3Editor(QWidget):
    valueChanged = Signal(object)

    def __init__(
        self,
        *,
        minimum: float = -1_000_000.0,
        maximum: float = 1_000_000.0,
        decimals: int = 4,
        step: float = 1.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.spins: list[QDoubleSpinBox] = []
        for label in ("X", "Y", "Z"):
            spin = _NoWheelDoubleSpinBox()
            spin.setRange(minimum, maximum)
            spin.setDecimals(decimals)
            spin.setSingleStep(step)
            spin.setPrefix(f"{label} ")
            spin.setKeyboardTracking(False)
            _make_horizontally_compact(spin, 60)
            spin.valueChanged.connect(self._emit_value)
            layout.addWidget(spin, 1)
            self.spins.append(spin)

    def value(self) -> np.ndarray:
        return np.asarray([spin.value() for spin in self.spins], dtype=float)

    def setValue(self, value: Iterable[float]) -> None:  # noqa: N802 - Qt naming
        values = list(value)
        if len(values) != 3:
            raise ValueError("Vector3Editor requires exactly three values")
        with _blocked(*self.spins):
            for spin, component in zip(self.spins, values, strict=True):
                spin.setValue(float(component))

    def _emit_value(self) -> None:
        self.valueChanged.emit(self.value())


class JointEditorWidget(QGroupBox):
    applyRequested = Signal(object)
    positionChanged = Signal(float)
    originFromSelectionRequested = Signal()
    axisPreviewRequested = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("선택 링크의 동작 설정", parent)
        self._joint: Any | None = None
        self._loading = False

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        # Labels above fields use the narrow inspector much more effectively
        # and keep the layout deterministic before the top-level window shows.
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)

        self.name_edit = QLineEdit()
        self.type_combo = _NoWheelComboBox()
        self.type_combo.addItems(
            ["fixed", "prismatic", "revolute", "continuous", "planar", "floating"]
        )
        self.parent_combo = _NoWheelComboBox()
        self.child_combo = _NoWheelComboBox()
        for combo in (self.type_combo, self.parent_combo, self.child_combo):
            # Arbitrarily long imported URDF names must not determine the
            # inspector's minimum width. The popup still exposes full entries.
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(10)
            _make_horizontally_compact(combo, 96)
        self.origin_editor = Vector3Editor(decimals=3, step=1.0)
        self.axis_editor = Vector3Editor(minimum=-1.0, maximum=1.0, decimals=4, step=0.1)
        self.axis_editor.setValue((1.0, 0.0, 0.0))

        form.addRow("이름", self.name_edit)
        form.addRow("동작 종류", self.type_combo)
        form.addRow("부모 링크", self.parent_combo)
        form.addRow("현재 링크", self.child_combo)
        self.parent_combo.setEnabled(False)
        self.child_combo.setEnabled(False)
        self.parent_combo.setToolTip("부모 관계는 왼쪽 링크 트리에서 구성합니다.")
        self.child_combo.setToolTip("현재 선택한 링크입니다.")

        origin_row = QWidget()
        origin_layout = QVBoxLayout(origin_row)
        origin_layout.setContentsMargins(0, 0, 0, 0)
        origin_layout.addWidget(self.origin_editor)
        self.origin_from_selection = QPushButton("선택 형상 중심 사용")
        self.origin_from_selection.clicked.connect(self.originFromSelectionRequested)
        origin_layout.addWidget(self.origin_from_selection)
        form.addRow("원점 (mm)", origin_row)

        axis_row = QWidget()
        axis_layout = QVBoxLayout(axis_row)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        axis_layout.addWidget(self.axis_editor)
        shortcut_layout = QGridLayout()
        shortcut_layout.setContentsMargins(0, 0, 0, 0)
        shortcut_layout.setHorizontalSpacing(5)
        shortcut_layout.setVerticalSpacing(4)
        for index, (text, vector) in enumerate(
            (
                ("+X", (1, 0, 0)),
                ("−X", (-1, 0, 0)),
                ("+Y", (0, 1, 0)),
                ("−Y", (0, -1, 0)),
                ("+Z", (0, 0, 1)),
                ("−Z", (0, 0, -1)),
            )
        ):
            button = QPushButton(text)
            button.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            button.clicked.connect(
                lambda _checked=False, v=vector: self._set_axis_and_preview(v)
            )
            shortcut_layout.addWidget(button, index % 2, index // 2)
        for column in range(3):
            shortcut_layout.setColumnStretch(column, 1)
        axis_layout.addLayout(shortcut_layout)
        self.flip_axis_button = QPushButton("방향 반전")
        self.flip_axis_button.setToolTip("현재 축의 부호를 반대로 바꿉니다.")
        self.flip_axis_button.clicked.connect(self._flip_axis)
        axis_layout.addWidget(self.flip_axis_button)
        form.addRow("이동/회전 방향", axis_row)

        limits = QWidget()
        limits_layout = QHBoxLayout(limits)
        limits_layout.setContentsMargins(0, 0, 0, 0)
        self.lower_spin = _NoWheelDoubleSpinBox()
        self.upper_spin = _NoWheelDoubleSpinBox()
        for spin in (self.lower_spin, self.upper_spin):
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setDecimals(3)
            spin.setKeyboardTracking(False)
            _make_horizontally_compact(spin)
        limits_layout.addWidget(QLabel("상태 0"))
        limits_layout.addWidget(self.lower_spin, 1)
        limits_layout.addWidget(QLabel("상태 1"))
        limits_layout.addWidget(self.upper_spin, 1)
        self.limit_units = QLabel("mm")
        limits_layout.addWidget(self.limit_units)
        form.addRow("0/1 실제값", limits)
        root.addLayout(form)

        self.apply_button = QPushButton("동작 설정 적용")
        self.apply_button.clicked.connect(self._emit_apply)
        root.addWidget(self.apply_button)

        self.preview_box = QGroupBox("선택 관절 동작 시험")
        preview_layout = QVBoxLayout(self.preview_box)
        position_grid = QGridLayout()
        position_grid.setContentsMargins(0, 0, 0, 0)
        position_grid.setHorizontalSpacing(5)
        position_grid.setVerticalSpacing(5)
        self.minus_button = QPushButton("−")
        self.plus_button = QPushButton("+")
        for button in (self.minus_button, self.plus_button):
            button.setFixedWidth(34)
        self.position_spin = _NoWheelDoubleSpinBox()
        self.step_spin = _NoWheelDoubleSpinBox()
        for spin in (self.position_spin, self.step_spin):
            spin.setDecimals(3)
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setKeyboardTracking(False)
            _make_horizontally_compact(spin)
        self.step_spin.setValue(5.0)
        self.position_units = QLabel("mm")
        self.step_units = QLabel("mm")
        position_grid.addWidget(QLabel("현재 실제값"), 0, 0)
        position_grid.addWidget(self.minus_button, 0, 1)
        position_grid.addWidget(self.position_spin, 0, 2)
        position_grid.addWidget(self.position_units, 0, 3)
        position_grid.addWidget(self.plus_button, 0, 4)
        position_grid.addWidget(QLabel("증감 간격"), 1, 0)
        position_grid.addWidget(self.step_spin, 1, 2)
        position_grid.addWidget(self.step_units, 1, 3)
        position_grid.setColumnStretch(2, 1)
        preview_layout.addLayout(position_grid)
        state_grid = QGridLayout()
        state_grid.setContentsMargins(0, 0, 0, 0)
        state_grid.setHorizontalSpacing(5)
        state_grid.setVerticalSpacing(5)
        state_grid.addWidget(QLabel("상태 0"), 0, 0)
        self.position_slider = _NoWheelSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        state_grid.addWidget(self.position_slider, 0, 1)
        state_grid.addWidget(QLabel("상태 1"), 0, 2)
        self.state_spin = _NoWheelDoubleSpinBox()
        self.state_spin.setRange(0.0, 1.0)
        self.state_spin.setDecimals(3)
        self.state_spin.setSingleStep(0.01)
        self.state_spin.setKeyboardTracking(False)
        self.state_spin.setPrefix("t=")
        self.state_spin.setToolTip("0과 1 사이의 정규화된 동작 상태입니다.")
        _make_horizontally_compact(self.state_spin, 72)
        state_grid.addWidget(QLabel("정규화 위치"), 1, 0)
        state_grid.addWidget(self.state_spin, 1, 1, 1, 2)
        state_grid.setColumnStretch(1, 1)
        preview_layout.addLayout(state_grid)
        # Keep motion testing above the detailed origin/axis form. The user can
        # select a moving link in the tree and reach the live slider without
        # scrolling through all advanced joint settings first.
        root.insertWidget(0, self.preview_box)

        self.type_combo.currentTextChanged.connect(self._refresh_units)
        self.lower_spin.valueChanged.connect(self._refresh_position_range)
        self.upper_spin.valueChanged.connect(self._refresh_position_range)
        self.position_spin.valueChanged.connect(self._spin_position_changed)
        self.position_slider.valueChanged.connect(self._slider_position_changed)
        self.state_spin.valueChanged.connect(self._state_position_changed)
        self.minus_button.clicked.connect(lambda: self._nudge(-1.0))
        self.plus_button.clicked.connect(lambda: self._nudge(1.0))
        self.setEnabled(False)

    def set_link_names(self, names: Iterable[str]) -> None:
        names = list(names)
        current_parent = self.parent_combo.currentText()
        current_child = self.child_combo.currentText()
        with _blocked(self.parent_combo, self.child_combo):
            self.parent_combo.clear()
            self.child_combo.clear()
            self.parent_combo.addItems(names)
            self.child_combo.addItems(names)
            if current_parent in names:
                self.parent_combo.setCurrentText(current_parent)
            if current_child in names:
                self.child_combo.setCurrentText(current_child)

    def set_joint(self, joint: Any | None) -> None:
        self._joint = joint
        self.setEnabled(joint is not None)
        if joint is None:
            return
        widgets = (
            self.name_edit,
            self.type_combo,
            self.parent_combo,
            self.child_combo,
            self.lower_spin,
            self.upper_spin,
            self.position_spin,
            self.position_slider,
            self.state_spin,
        )
        self._loading = True
        try:
            with _blocked(*widgets):
                self.name_edit.setText(str(getattr(joint, "name", "joint")))
                self.type_combo.setCurrentText(str(getattr(joint, "type", "fixed")))
                self.parent_combo.setCurrentText(str(getattr(joint, "parent", "")))
                self.child_combo.setCurrentText(str(getattr(joint, "child", "")))
                self.origin_editor.setValue(np.asarray(getattr(joint, "origin_xyz", (0, 0, 0))) * 1000.0)
                self.axis_editor.setValue(getattr(joint, "axis", (1, 0, 0)))
                scale = self._joint_display_scale()
                lower = getattr(joint, "lower", None)
                upper = getattr(joint, "upper", None)
                if lower is None:
                    lower = -math.pi if getattr(joint, "type", "") == "continuous" else 0.0
                if upper is None:
                    upper = math.pi if getattr(joint, "type", "") == "continuous" else 0.0
                self.lower_spin.setValue(float(lower) * scale)
                self.upper_spin.setValue(float(upper) * scale)
                # Refresh the range before loading the position. Otherwise a
                # value can be silently clamped by the previously shown
                # joint's narrower range when switching joint types.
                self._refresh_units()
                self.position_spin.setValue(
                    float(getattr(joint, "position", 0.0)) * scale
                )
                self._sync_slider_from_spin()
        finally:
            self._loading = False

    def current_values(self) -> dict[str, Any]:
        joint_type = self.type_combo.currentText()
        scale = self._joint_display_scale(joint_type)
        axis = self.axis_editor.value()
        norm = float(np.linalg.norm(axis))
        if norm < 1.0e-9:
            raise ValueError("관절축은 0 벡터일 수 없습니다.")
        axis /= norm
        lower = self.lower_spin.value() / scale
        upper = self.upper_spin.value() / scale
        if joint_type == "continuous":
            lower, upper = -math.pi, math.pi
        elif joint_type in {"fixed", "planar", "floating"}:
            lower, upper = 0.0, 0.0
        elif lower > upper:
            raise ValueError(
                "상태 0 실제값은 상태 1 실제값보다 클 수 없습니다. 반대 방향은 −축을 사용하세요."
            )
        return {
            "name": self.name_edit.text().strip(),
            "type": joint_type,
            "parent": self.parent_combo.currentText(),
            "child": self.child_combo.currentText(),
            "origin_xyz": self.origin_editor.value() / 1000.0,
            "origin_rpy": np.asarray(getattr(self._joint, "origin_rpy", (0, 0, 0)), dtype=float),
            "axis": axis,
            "lower": lower,
            "upper": upper,
            "position": self.position_spin.value() / scale,
        }

    def set_origin_mm(self, value: Iterable[float]) -> None:
        self.origin_editor.setValue(value)

    def set_axis(self, value: Iterable[float]) -> None:
        self.axis_editor.setValue(value)

    def _flip_axis(self) -> None:
        self._set_axis_and_preview(-self.axis_editor.value())

    def _set_axis_and_preview(self, value: Iterable[float]) -> None:
        self.axis_editor.setValue(value)
        self.axisPreviewRequested.emit(self.axis_editor.value())

    def _joint_display_scale(self, joint_type: str | None = None) -> float:
        kind = joint_type or self.type_combo.currentText()
        return 1000.0 if kind == "prismatic" else 180.0 / math.pi

    def _refresh_units(self) -> None:
        kind = self.type_combo.currentText()
        units = "mm" if kind == "prismatic" else "deg"
        if kind in {"fixed", "planar", "floating"}:
            units = "—"
        self.limit_units.setText(units)
        self.position_units.setText(units)
        self.step_units.setText(units)
        self.preview_box.setEnabled(kind in {"prismatic", "revolute", "continuous"})
        scalar = kind in {"prismatic", "revolute", "continuous"}
        self.flip_axis_button.setEnabled(scalar)
        self._refresh_position_range()

    def _refresh_position_range(self) -> None:
        low = self.lower_spin.value()
        high = self.upper_spin.value()
        if low > high:
            low, high = high, low
        if math.isclose(low, high):
            high = low + 1.0
        with _blocked(self.position_spin):
            self.position_spin.setRange(low, high)
            self.position_spin.setValue(min(max(self.position_spin.value(), low), high))
        self._sync_slider_from_spin()

    def _sync_slider_from_spin(self) -> None:
        low = self.position_spin.minimum()
        high = self.position_spin.maximum()
        ratio = 0.0 if math.isclose(low, high) else (self.position_spin.value() - low) / (high - low)
        ratio = max(0.0, min(1.0, ratio))
        with _blocked(self.position_slider, self.state_spin):
            self.position_slider.setValue(int(round(ratio * 1000.0)))
            self.state_spin.setValue(ratio)

    def _spin_position_changed(self, display_value: float) -> None:
        if self._loading:
            return
        self._sync_slider_from_spin()
        self.positionChanged.emit(display_value / self._joint_display_scale())

    def _slider_position_changed(self, slider_value: int) -> None:
        low = self.position_spin.minimum()
        high = self.position_spin.maximum()
        ratio = slider_value / 1000.0
        value = low + (high - low) * ratio
        with _blocked(self.position_spin, self.state_spin):
            self.position_spin.setValue(value)
            self.state_spin.setValue(ratio)
        if not self._loading:
            self.positionChanged.emit(value / self._joint_display_scale())

    def _state_position_changed(self, ratio: float) -> None:
        slider_value = int(round(max(0.0, min(1.0, ratio)) * 1000.0))
        low = self.position_spin.minimum()
        high = self.position_spin.maximum()
        value = low + (high - low) * slider_value / 1000.0
        with _blocked(self.position_slider, self.position_spin):
            self.position_slider.setValue(slider_value)
            self.position_spin.setValue(value)
        if not self._loading:
            self.positionChanged.emit(value / self._joint_display_scale())

    def _nudge(self, direction: float) -> None:
        self.position_spin.setValue(self.position_spin.value() + direction * abs(self.step_spin.value()))

    def _emit_apply(self) -> None:
        try:
            values = self.current_values()
        except ValueError as exc:
            QMessageBox.warning(self, "관절 설정 오류", str(exc))
            return
        self.applyRequested.emit(values)
