from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Iterable

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
    motionPreviewRequested = Signal()
    axisEditModeChanged = Signal(bool)

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
        self._suggested_origin: np.ndarray | None = None
        self._axis_candidates: dict[str, np.ndarray] = {}
        self._axis_candidate_origins: dict[str, np.ndarray] = {}
        self._axis_candidate_descriptions: dict[str, str] = {}
        self._preferred_candidate: str | None = None
        self._selected_axis_name: str | None = None
        self.setWindowTitle("자식 링크와 동작 만들기 · 3D 조작 가능")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.Tool, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setMinimumWidth(420)
        self.setMinimumHeight(520)
        outer = QVBoxLayout(self)
        self._content_scroll = QScrollArea()
        self._content_scroll.setWidgetResizable(True)
        self._content_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._content_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        content = QWidget()
        root = QVBoxLayout(content)
        self._content_scroll.setWidget(content)
        outer.addWidget(self._content_scroll, 1)
        summary_text = (
            f"선택한 자식 형상 {selected_count}개를 하나의 강체 링크로 만들고, "
            "부모 위에서 회전축과 움직임을 확인합니다. 이 창이 열린 상태에서도 "
            "3D 화면을 회전·확대·이동할 수 있습니다."
        )
        summary = QLabel(summary_text)
        summary.setWordWrap(True)
        root.addWidget(summary)
        self.preset_hint = QLabel()
        self.preset_hint.setWordWrap(True)
        self.preset_hint.setStyleSheet(
            "QLabel { color: #3f5870; background: #eef5fb; "
            "border: 1px solid #c9d9e6; border-radius: 4px; padding: 7px; }"
        )
        self.preset_hint.setVisible(False)
        root.addWidget(self.preset_hint)

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
        self.origin_editor = Vector3Editor(decimals=3, step=0.1)
        self.axis_editor = Vector3Editor(
            minimum=-1.0,
            maximum=1.0,
            decimals=4,
            step=0.01,
        )
        self.axis_editor.setValue((1.0, 0.0, 0.0))

        origin_box = QWidget()
        origin_layout = QVBoxLayout(origin_box)
        origin_layout.setContentsMargins(0, 0, 0, 0)
        origin_hint = QLabel(
            "핸들이 도는 축이 통과하는 점입니다. 3D의 노란 점과 ‘중심’ 표시가 이 값입니다."
        )
        origin_hint.setWordWrap(True)
        origin_layout.addWidget(origin_hint)
        origin_layout.addWidget(self.origin_editor)
        self.origin_reset_button = QPushButton("자동으로 찾은 중심으로 복원")
        self.origin_reset_button.setToolTip(
            "선택한 자식 형상의 경계 중심으로 관절 중심을 되돌립니다."
        )
        origin_layout.addWidget(self.origin_reset_button)

        axis_box = QWidget()
        axis_layout = QVBoxLayout(axis_box)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        geometry_shortcuts = QHBoxLayout()
        self.geometry_axis_buttons: dict[str, QPushButton] = {}
        for name, label, color in (
            ("A", "A  장축\n가장 긴 방향", "#dc4943"),
            ("B", "B  중간축\n두 번째 방향", "#38b95a"),
            ("C", "C  면의 노멀\n회전 추천", "#4b7fe8"),
        ):
            button = QPushButton(label)
            button.setEnabled(False)
            button.setCheckable(True)
            button.setStyleSheet(
                f"QPushButton {{ border: 2px solid {color}; padding: 6px; }}"
                f"QPushButton:checked {{ background: {color}; color: white; font-weight: 700; }}"
            )
            button.clicked.connect(
                lambda _checked=False, candidate=name: self.select_candidate_axis(candidate)
            )
            geometry_shortcuts.addWidget(button)
            self.geometry_axis_buttons[name] = button
        axis_layout.addLayout(geometry_shortcuts)
        self.selected_axis_label = QLabel("3D의 A/B/C 중 하나를 선택하세요.")
        self.selected_axis_label.setWordWrap(True)
        axis_layout.addWidget(self.selected_axis_label)
        self.axis_direction_button = QPushButton("↻  회전 방향 반전")
        self.axis_direction_button.setMinimumHeight(34)
        self.axis_direction_button.setToolTip(
            "3D의 주황색 원형 화살표 방향을 반대로 바꿉니다."
        )
        axis_layout.addWidget(self.axis_direction_button)
        self.axis_edit_button = QPushButton("3D에서 축 위치·방향 직접 조정")
        self.axis_edit_button.setCheckable(True)
        self.axis_edit_button.setMinimumHeight(36)
        self.axis_edit_button.setToolTip(
            "켜면 3D에 노란 조절선이 나타납니다. 양 끝점을 끌면 방향이, "
            "선 가운데를 끌면 관절 중심이 바뀝니다."
        )
        axis_layout.addWidget(self.axis_edit_button)
        axis_hint = QLabel(
            "3D의 색 선이나 위 버튼을 선택하세요. 노란 점은 관절 중심, "
            "주황색 원형 화살표는 실제 양의 회전 방향입니다. 직접 조정을 켜면 "
            "노란 선의 양 끝점으로 축을 미세 조정할 수 있습니다."
        )
        axis_hint.setWordWrap(True)
        axis_layout.addWidget(axis_hint)

        self.advanced_axis_button = QPushButton("고급: 축 벡터 직접 입력 펼치기")
        self.advanced_axis_button.setCheckable(True)
        axis_layout.addWidget(self.advanced_axis_button)
        self.advanced_axis_box = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_axis_box)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.addWidget(self.axis_editor)
        manual_shortcuts = QHBoxLayout()
        for label, vector, color in (
            ("X축", (1, 0, 0), "#dc4943"),
            ("Y축", (0, 1, 0), "#38b95a"),
            ("Z축", (0, 0, 1), "#4b7fe8"),
        ):
            button = QPushButton(label)
            button.setStyleSheet(f"QPushButton {{ border: 2px solid {color}; }}")
            button.clicked.connect(
                lambda _checked=False, value=vector: self._choose_candidate_axis(value)
            )
            manual_shortcuts.addWidget(button)
        advanced_layout.addLayout(manual_shortcuts)
        self.advanced_axis_box.setVisible(False)
        axis_layout.addWidget(self.advanced_axis_box)

        preview_box = QWidget()
        preview_layout = QHBoxLayout(preview_box)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_slider = _NoWheelSlider(Qt.Orientation.Horizontal)
        self.preview_slider.setRange(-100, 100)
        self.preview_slider.setValue(0)
        self.preview_value_label = QLabel("0°")
        self.preview_value_label.setMinimumWidth(58)
        self.preview_reset_button = QPushButton("0 위치")
        self.preview_reset_button.setToolTip(
            "관절 중심과 축 설정은 유지하고 동작 미리보기만 초기 위치로 되돌립니다."
        )
        preview_layout.addWidget(self.preview_slider, 1)
        preview_layout.addWidget(self.preview_value_label)
        preview_layout.addWidget(self.preview_reset_button)

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
        self.lower_state_label = QLabel("상태 0")
        self.upper_state_label = QLabel("상태 1")
        limit_layout.addWidget(self.lower_state_label)
        limit_layout.addWidget(self.lower_spin)
        limit_layout.addWidget(self.upper_state_label)
        limit_layout.addWidget(self.upper_spin)
        limit_layout.addWidget(self.units_label)

        form.addRow("링크 이름", self.name_edit)
        form.addRow("부모 링크", self.parent_combo)
        form.addRow("동작 종류", self.type_combo)
        form.addRow("1. 관절 중심 (mm)", origin_box)
        form.addRow("2. 회전/이동 축 선택", axis_box)
        form.addRow("동작 미리보기", preview_box)
        form.addRow("0/1 실제값", limit_box)
        root.addLayout(form)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)
        self.type_combo.currentTextChanged.connect(self._refresh_type)
        self.parent_combo.currentTextChanged.connect(self.parentPreviewRequested.emit)
        self.origin_reset_button.clicked.connect(self.reset_candidate_origin)
        self.origin_editor.valueChanged.connect(self._origin_value_changed)
        self.axis_editor.valueChanged.connect(self._axis_value_changed)
        self.axis_direction_button.clicked.connect(self.flip_axis_direction)
        self.axis_edit_button.toggled.connect(self._toggle_axis_edit_mode)
        self.advanced_axis_button.toggled.connect(self._toggle_advanced_axis)
        self.preview_slider.valueChanged.connect(self._preview_slider_changed)
        self.preview_reset_button.clicked.connect(self.reset_motion_preview)
        self.lower_spin.valueChanged.connect(self._limits_changed)
        self.upper_spin.valueChanged.connect(self._limits_changed)
        self._refresh_type(self.type_combo.currentText())

    def apply_mechanism_preset(self, values: dict[str, Any]) -> None:
        """Use wizard output as editable defaults for precise 3D authoring."""

        self.preset_hint.setText(
            f"마법사: {values.get('title', '대표 기구')} · "
            "아래 값은 시작값입니다. 3D에서 축과 중심을 확인한 뒤 필요하면 수정하세요."
        )
        self.preset_hint.setVisible(True)
        self.name_edit.setText(str(values.get("link_name") or "moving_link"))
        parent = str(values.get("parent") or "")
        if parent:
            self.parent_combo.setCurrentText(parent)
        self.type_combo.setCurrentText(str(values.get("joint_type") or "prismatic"))
        self.lower_spin.setValue(float(values.get("lower", 0.0)))
        self.upper_spin.setValue(float(values.get("upper", 100.0)))
        self.lower_state_label.setText(str(values.get("state_0") or "상태 0"))
        self.upper_state_label.setText(str(values.get("state_1") or "상태 1"))
        self.reset_motion_preview()

    def set_candidate_origin(self, value: Iterable[float]) -> None:
        origin = np.asarray(tuple(value), dtype=float)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("Candidate origin must contain three finite values")
        self._suggested_origin = origin.copy()
        self._candidate_origin = origin
        self.origin_editor.setValue(origin * 1000.0)

    def reset_candidate_origin(self) -> None:
        if self._suggested_origin is None:
            return
        self._candidate_origin = self._suggested_origin.copy()
        self.origin_editor.setValue(self._suggested_origin * 1000.0)
        self.motionPreviewRequested.emit()

    def set_axis_candidates(
        self,
        values: dict[str, Iterable[float]],
        *,
        origins: dict[str, Iterable[float]] | None = None,
        descriptions: dict[str, str] | None = None,
        recommended: str | None = None,
    ) -> None:
        candidates: dict[str, np.ndarray] = {}
        for raw_name, raw_value in values.items():
            name = str(raw_name).upper().lstrip("+-")
            vector = np.asarray(tuple(raw_value), dtype=float)
            norm = float(np.linalg.norm(vector))
            if name and vector.shape == (3,) and np.all(np.isfinite(vector)) and norm > 1.0e-12:
                candidates[name] = vector / norm
        self._axis_candidates = candidates
        self._axis_candidate_origins = {}
        for raw_name, raw_value in (origins or {}).items():
            name = str(raw_name).upper().lstrip("+-")
            origin = np.asarray(tuple(raw_value), dtype=float)
            if name in candidates and origin.shape == (3,) and np.all(np.isfinite(origin)):
                self._axis_candidate_origins[name] = origin
        self._axis_candidate_descriptions = {
            str(name).upper().lstrip("+-"): str(description)
            for name, description in (descriptions or {}).items()
        }
        self._preferred_candidate = (
            str(recommended).upper().lstrip("+-")
            if recommended is not None
            else None
        )
        default_labels = {
            "A": "A  장축\n가장 긴 방향",
            "B": "B  중간축\n두 번째 방향",
            "C": "C  면의 노멀\n회전 추천",
        }
        for name, button in self.geometry_axis_buttons.items():
            button.setEnabled(name in candidates)
            button.setText(
                self._axis_candidate_descriptions.get(name, default_labels[name])
            )
        preferred = self._preferred_candidate
        if preferred is None:
            preferred = (
                "C"
                if self.type_combo.currentText() in {"revolute", "continuous"}
                else "A"
            )
        if preferred in candidates:
            self.select_candidate_axis(preferred)

    def candidate_axes(self) -> dict[str, np.ndarray]:
        return {name: value.copy() for name, value in self._axis_candidates.items()}

    def candidate_origins(self) -> dict[str, np.ndarray]:
        return {
            name: value.copy()
            for name, value in self._axis_candidate_origins.items()
        }

    def matching_candidate(self, value: Iterable[float]) -> str | None:
        vector = np.asarray(tuple(value), dtype=float)
        norm = float(np.linalg.norm(vector))
        if vector.shape != (3,) or not np.all(np.isfinite(vector)) or norm <= 1.0e-12:
            return None
        unit = vector / norm
        if not self._axis_candidates:
            return None
        name, score = max(
            (
                (candidate_name, abs(float(np.dot(unit, candidate))))
                for candidate_name, candidate in self._axis_candidates.items()
            ),
            key=lambda item: item[1],
        )
        return name if score >= 0.995 else None

    def select_candidate_axis(self, name: str) -> None:
        vectors: dict[str, Iterable[float]] = {
            "X": (1.0, 0.0, 0.0),
            "Y": (0.0, 1.0, 0.0),
            "Z": (0.0, 0.0, 1.0),
        }
        vectors.update(self._axis_candidates)
        axis = str(name).upper().lstrip("+-")
        if axis in vectors:
            self._selected_axis_name = axis
            if axis in self._axis_candidate_origins:
                candidate_origin = self._axis_candidate_origins[axis]
                self._suggested_origin = candidate_origin.copy()
                self._candidate_origin = candidate_origin.copy()
                self.origin_editor.setValue(candidate_origin * 1000.0)
            self._choose_candidate_axis(vectors[axis])

    def _choose_candidate_axis(self, value: Iterable[float]) -> None:
        self.axis_editor.setValue(value)
        self._update_axis_choice_state()
        self.axisPreviewRequested.emit(self.axis_editor.value())
        self.motionPreviewRequested.emit()

    def flip_axis_direction(self) -> None:
        value = self.axis_editor.value()
        if np.linalg.norm(value) <= 1.0e-12:
            return
        self._choose_candidate_axis(-value)

    def _toggle_axis_edit_mode(self, enabled: bool) -> None:
        self.axis_edit_button.setText(
            "3D 축 직접 조정 종료"
            if enabled
            else "3D에서 축 위치·방향 직접 조정"
        )
        self.axisEditModeChanged.emit(enabled)

    def set_axis_from_3d(
        self,
        origin_parent: Iterable[float],
        axis_parent: Iterable[float],
    ) -> None:
        """Apply a world-handle edit converted into the selected parent frame."""

        origin = np.asarray(tuple(origin_parent), dtype=float)
        axis = np.asarray(tuple(axis_parent), dtype=float)
        magnitude = float(np.linalg.norm(axis))
        if (
            origin.shape != (3,)
            or axis.shape != (3,)
            or not np.all(np.isfinite(origin))
            or not np.all(np.isfinite(axis))
            or magnitude <= 1.0e-12
        ):
            return
        axis /= magnitude
        self._candidate_origin = origin.copy()
        with _blocked(self.origin_editor, self.axis_editor):
            self.origin_editor.setValue(origin * 1000.0)
            self.axis_editor.setValue(axis)
        self._selected_axis_name = self.matching_candidate(axis)
        self._update_axis_choice_state()
        self.axisPreviewRequested.emit(axis.copy())
        self.motionPreviewRequested.emit()

    def _toggle_advanced_axis(self, expanded: bool) -> None:
        self.advanced_axis_box.setVisible(expanded)
        self.advanced_axis_button.setText(
            "고급: 축 벡터 직접 입력 접기"
            if expanded
            else "고급: 축 벡터 직접 입력 펼치기"
        )
        self.fit_to_available_screen()
        if expanded:
            QTimer.singleShot(
                0,
                lambda: self._content_scroll.ensureWidgetVisible(
                    self.advanced_axis_box,
                    12,
                    12,
                ),
            )

    def fit_to_available_screen(self) -> None:
        screen = self.screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        width = min(max(self.sizeHint().width(), 440), available.width() - 32)
        height = min(max(self.sizeHint().height(), 620), available.height() - 56)
        self.resize(width, height)

    def _update_axis_choice_state(self) -> None:
        value = self.axis_editor.value()
        match = self.matching_candidate(value)
        for name, button in self.geometry_axis_buttons.items():
            button.setChecked(name == match)
        descriptions = {
            "A": "A 장축 · 형상에서 가장 긴 방향",
            "B": "B 중간축 · 형상에서 두 번째로 긴 방향",
            "C": "C 면의 노멀 · 평면형 핸들의 추천 회전축",
        }
        descriptions.update(self._axis_candidate_descriptions)
        if match is not None:
            direction = self._axis_candidates[match]
            sign = "정방향" if float(np.dot(value, direction)) >= 0.0 else "반대 방향"
            self.selected_axis_label.setText(f"선택: {descriptions[match]} · {sign}")
        else:
            self.selected_axis_label.setText("선택: 직접 입력한 사용자 축")
        angular = self.type_combo.currentText() in {"revolute", "continuous"}
        self.axis_direction_button.setText(
            "↻  주황색 회전 방향 반전"
            if angular
            else "⇄  이동 방향 반전"
        )

    def _origin_value_changed(self, value: Iterable[float]) -> None:
        self._candidate_origin = np.asarray(tuple(value), dtype=float) / 1000.0
        self.motionPreviewRequested.emit()

    def _axis_value_changed(self, value: Iterable[float]) -> None:
        self._selected_axis_name = self.matching_candidate(value)
        self._update_axis_choice_state()
        self.axisPreviewRequested.emit(value)
        self.motionPreviewRequested.emit()

    def _preview_slider_changed(self, _value: int) -> None:
        kind = self.type_combo.currentText()
        position = self.preview_position_si()
        if kind in {"revolute", "continuous"}:
            self.preview_value_label.setText(f"{math.degrees(position):.0f}°")
        elif kind == "prismatic":
            self.preview_value_label.setText(f"{position * 1000.0:.0f} mm")
        else:
            self.preview_value_label.setText("—")
        self.motionPreviewRequested.emit()

    def preview_position_si(self) -> float:
        fraction = self.preview_slider.value() / 100.0
        kind = self.type_combo.currentText()
        if kind == "fixed":
            return 0.0

        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        if lower <= 0.0 <= upper:
            # Keep the slider center as the URDF zero pose while allowing each
            # side to reach its actual (possibly asymmetric) joint limit.
            display_value = (
                fraction * upper
                if fraction >= 0.0
                else (-fraction) * lower
            )
        else:
            # If zero is outside the legal interval, use the whole slider for
            # a conventional state-0 to state-1 interpolation.
            interpolation = (fraction + 1.0) * 0.5
            display_value = lower + (upper - lower) * interpolation

        if kind in {"revolute", "continuous"}:
            return math.radians(display_value)
        if kind == "prismatic":
            return display_value / 1000.0
        return 0.0

    def _limits_changed(self, _value: float) -> None:
        self._preview_slider_changed(self.preview_slider.value())

    def reset_motion_preview(self) -> None:
        lower = self.lower_spin.value()
        upper = self.upper_spin.value()
        if lower <= 0.0 <= upper:
            slider_value = 0
        elif 0.0 < lower <= upper:
            slider_value = -100
        elif lower <= upper < 0.0:
            slider_value = 100
        else:
            slider_value = 0
        if self.preview_slider.value() == slider_value:
            self._preview_slider_changed(slider_value)
        else:
            self.preview_slider.setValue(slider_value)

    def _refresh_type(self, joint_type: str) -> None:
        scalar = joint_type in {"prismatic", "revolute", "continuous"}
        self.axis_editor.setEnabled(scalar)
        self.lower_spin.setEnabled(joint_type != "fixed")
        self.upper_spin.setEnabled(joint_type != "fixed")
        angular = joint_type in {"revolute", "continuous"}
        self.units_label.setText("deg" if angular else ("mm" if joint_type == "prismatic" else "—"))
        self.preview_slider.setEnabled(joint_type != "fixed")
        self.axis_direction_button.setEnabled(scalar)
        self.axis_edit_button.setEnabled(scalar)
        if not scalar and self.axis_edit_button.isChecked():
            self.axis_edit_button.setChecked(False)
        if joint_type == "continuous":
            self.lower_spin.setValue(-180.0)
            self.upper_spin.setValue(180.0)
        recommended = self._preferred_candidate or ("C" if angular else "A")
        if recommended in self._axis_candidates:
            self.select_candidate_axis(recommended)
        else:
            self._update_axis_choice_state()
        self._preview_slider_changed(self.preview_slider.value())

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
            "origin_xyz": self.origin_editor.value() / 1000.0,
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
        self._joint_specs: dict[str, Any] = {}
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

        physics_box = QGroupBox("시뮬레이션 물리 기본값")
        physics_form = QFormLayout(physics_box)
        physics_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.effort_spin = _NoWheelDoubleSpinBox()
        self.effort_spin.setRange(0.0, 1_000_000_000.0)
        self.effort_spin.setDecimals(3)
        self.effort_spin.setValue(100.0)
        self.velocity_spin = _NoWheelDoubleSpinBox()
        self.velocity_spin.setRange(0.0, 1_000_000.0)
        self.velocity_spin.setDecimals(3)
        self.velocity_spin.setValue(100.0)
        velocity_row = QWidget()
        velocity_layout = QHBoxLayout(velocity_row)
        velocity_layout.setContentsMargins(0, 0, 0, 0)
        velocity_layout.addWidget(self.velocity_spin, 1)
        self.velocity_units = QLabel("mm/s")
        velocity_layout.addWidget(self.velocity_units)
        self.damping_spin = _NoWheelDoubleSpinBox()
        self.damping_spin.setRange(0.0, 1_000_000.0)
        self.damping_spin.setDecimals(6)
        self.damping_spin.setSingleStep(0.01)
        self.friction_spin = _NoWheelDoubleSpinBox()
        self.friction_spin.setRange(0.0, 1_000_000.0)
        self.friction_spin.setDecimals(6)
        self.friction_spin.setSingleStep(0.01)
        for spin in (
            self.effort_spin,
            self.velocity_spin,
            self.damping_spin,
            self.friction_spin,
        ):
            spin.setKeyboardTracking(False)
            _make_horizontally_compact(spin)
        physics_form.addRow("최대 힘/토크 (effort)", self.effort_spin)
        physics_form.addRow("최대 속도", velocity_row)
        physics_form.addRow("감쇠 (damping)", self.damping_spin)
        physics_form.addRow("마찰 (friction)", self.friction_spin)
        physics_hint = QLabel(
            "관절의 URDF <limit>와 <dynamics>로 저장됩니다. 실제 장비 값이 있으면 "
            "제조사 사양으로 교체하세요."
        )
        physics_hint.setWordWrap(True)
        physics_form.addRow(physics_hint)
        form.addRow("물리/제어 제한", physics_box)

        mimic_box = QGroupBox("다른 관절과 연동 (mimic)")
        mimic_layout = QVBoxLayout(mimic_box)
        mimic_layout.setContentsMargins(8, 8, 8, 8)
        self.mimic_enable = QCheckBox("이 관절을 다른 관절에 따라 움직이기")
        mimic_layout.addWidget(self.mimic_enable)
        mimic_form = QFormLayout()
        mimic_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.mimic_source_combo = _NoWheelComboBox()
        self.mimic_source_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.mimic_source_combo.setMinimumContentsLength(10)
        _make_horizontally_compact(self.mimic_source_combo, 96)
        mimic_form.addRow("구동 관절", self.mimic_source_combo)
        self.mimic_auto_check = QCheckBox("상태 0↔1에서 배율 자동 계산")
        self.mimic_auto_check.setChecked(True)
        self.mimic_auto_check.setToolTip(
            "구동 관절과 이 관절의 상태 0/1 실제값을 대응시켜 "
            "m/rad 등의 배율을 자동 계산합니다."
        )
        mimic_form.addRow("계산", self.mimic_auto_check)
        self.mimic_reverse_check = QCheckBox("반대 방향으로 연동")
        mimic_form.addRow("방향", self.mimic_reverse_check)
        self.mimic_multiplier_spin = _NoWheelDoubleSpinBox()
        self.mimic_multiplier_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.mimic_multiplier_spin.setDecimals(6)
        self.mimic_multiplier_spin.setSingleStep(0.01)
        self.mimic_multiplier_spin.setValue(1.0)
        _make_horizontally_compact(self.mimic_multiplier_spin)
        mimic_form.addRow("고급 배율", self.mimic_multiplier_spin)
        offset_row = QWidget()
        offset_layout = QHBoxLayout(offset_row)
        offset_layout.setContentsMargins(0, 0, 0, 0)
        self.mimic_offset_spin = _NoWheelDoubleSpinBox()
        self.mimic_offset_spin.setRange(-1_000_000.0, 1_000_000.0)
        self.mimic_offset_spin.setDecimals(3)
        self.mimic_offset_spin.setSingleStep(1.0)
        _make_horizontally_compact(self.mimic_offset_spin)
        self.mimic_offset_units = QLabel("mm")
        offset_layout.addWidget(self.mimic_offset_spin, 1)
        offset_layout.addWidget(self.mimic_offset_units)
        mimic_form.addRow("고급 오프셋", offset_row)
        mimic_layout.addLayout(mimic_form)
        self.mimic_formula_label = QLabel("연동을 켜면 자동 계산 결과가 표시됩니다.")
        self.mimic_formula_label.setWordWrap(True)
        self.mimic_formula_label.setStyleSheet(
            "QLabel { color: #3f5870; background: #eef5fb; "
            "border: 1px solid #c9d9e6; border-radius: 4px; padding: 5px; }"
        )
        mimic_layout.addWidget(self.mimic_formula_label)
        form.addRow("관절 연동", mimic_box)

        drive_box = QGroupBox("전·후진 레버로 바퀴 속도 구동")
        drive_layout = QVBoxLayout(drive_box)
        drive_layout.setContentsMargins(8, 8, 8, 8)
        self.drive_enable = QCheckBox("이 연속 회전 관절을 레버로 구동")
        self.drive_enable.setToolTip(
            "바퀴 관절에서 사용합니다. 레버 상태 0은 최대 후진, "
            "중앙은 정지, 상태 1은 최대 전진입니다."
        )
        drive_layout.addWidget(self.drive_enable)
        drive_form = QFormLayout()
        drive_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.drive_source_combo = _NoWheelComboBox()
        self.drive_source_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.drive_source_combo.setMinimumContentsLength(10)
        _make_horizontally_compact(self.drive_source_combo, 96)
        drive_form.addRow("전·후진 레버", self.drive_source_combo)
        rpm_row = QWidget()
        rpm_layout = QHBoxLayout(rpm_row)
        rpm_layout.setContentsMargins(0, 0, 0, 0)
        self.drive_max_rpm_spin = _NoWheelDoubleSpinBox()
        self.drive_max_rpm_spin.setRange(0.1, 100_000.0)
        self.drive_max_rpm_spin.setDecimals(1)
        self.drive_max_rpm_spin.setSingleStep(10.0)
        self.drive_max_rpm_spin.setValue(60.0)
        _make_horizontally_compact(self.drive_max_rpm_spin)
        rpm_layout.addWidget(self.drive_max_rpm_spin, 1)
        rpm_layout.addWidget(QLabel("RPM"))
        drive_form.addRow("상태 0/1 최대 속도", rpm_row)
        deadband_row = QWidget()
        deadband_layout = QHBoxLayout(deadband_row)
        deadband_layout.setContentsMargins(0, 0, 0, 0)
        self.drive_deadband_spin = _NoWheelDoubleSpinBox()
        self.drive_deadband_spin.setRange(0.0, 49.0)
        self.drive_deadband_spin.setDecimals(1)
        self.drive_deadband_spin.setSingleStep(1.0)
        self.drive_deadband_spin.setValue(3.0)
        _make_horizontally_compact(self.drive_deadband_spin)
        deadband_layout.addWidget(self.drive_deadband_spin, 1)
        deadband_layout.addWidget(QLabel("%"))
        drive_form.addRow("중립 범위", deadband_row)
        self.drive_reverse_check = QCheckBox("바퀴 회전 방향 반전")
        drive_form.addRow("방향", self.drive_reverse_check)
        drive_layout.addLayout(drive_form)
        self.drive_formula_label = QLabel(
            "바퀴 관절을 continuous로 바꾸면 속도 구동을 설정할 수 있습니다."
        )
        self.drive_formula_label.setWordWrap(True)
        self.drive_formula_label.setStyleSheet(
            "QLabel { color: #59440b; background: #fff8df; "
            "border: 1px solid #e6d39b; border-radius: 4px; padding: 5px; }"
        )
        drive_layout.addWidget(self.drive_formula_label)
        form.addRow("주행 구동", drive_box)
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
        self.lower_spin.valueChanged.connect(self._refresh_mimic_controls)
        self.upper_spin.valueChanged.connect(self._refresh_mimic_controls)
        self.mimic_enable.toggled.connect(self._refresh_mimic_controls)
        self.mimic_source_combo.currentTextChanged.connect(
            self._refresh_mimic_controls
        )
        self.mimic_auto_check.toggled.connect(self._refresh_mimic_controls)
        self.mimic_reverse_check.toggled.connect(self._refresh_mimic_controls)
        self.drive_enable.toggled.connect(self._refresh_drive_controls)
        self.drive_source_combo.currentTextChanged.connect(
            self._refresh_drive_controls
        )
        self.drive_max_rpm_spin.valueChanged.connect(self._refresh_drive_controls)
        self.drive_deadband_spin.valueChanged.connect(self._refresh_drive_controls)
        self.drive_reverse_check.toggled.connect(self._refresh_drive_controls)
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

    def set_joint_specs(self, joints: Iterable[Any]) -> None:
        """Provide available source joints for coupling selectors."""

        self._joint_specs = {
            str(getattr(joint, "name", "")): joint
            for joint in joints
            if str(getattr(joint, "name", ""))
        }
        self._refresh_mimic_sources()
        self._refresh_drive_sources()

    def _refresh_mimic_sources(self) -> None:
        current = self.mimic_source_combo.currentText()
        selected_name = str(getattr(self._joint, "name", ""))
        names = [
            name
            for name, joint in self._joint_specs.items()
            if name != selected_name
            and str(getattr(joint, "type", ""))
            in {"prismatic", "revolute", "continuous"}
        ]
        with _blocked(self.mimic_source_combo):
            self.mimic_source_combo.clear()
            self.mimic_source_combo.addItem("연동 안 함", None)
            for name in names:
                self.mimic_source_combo.addItem(name, name)
            if current in names:
                self.mimic_source_combo.setCurrentText(current)

    def _refresh_drive_sources(self) -> None:
        current = self.drive_source_combo.currentText()
        selected_name = str(getattr(self._joint, "name", ""))
        names = [
            name
            for name, joint in self._joint_specs.items()
            if name != selected_name
            and str(getattr(joint, "type", "")) in {"prismatic", "revolute"}
        ]
        with _blocked(self.drive_source_combo):
            self.drive_source_combo.clear()
            self.drive_source_combo.addItem("레버 선택 안 함", None)
            for name in names:
                self.drive_source_combo.addItem(name, name)
            if current in names:
                self.drive_source_combo.setCurrentText(current)

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
            self.effort_spin,
            self.velocity_spin,
            self.damping_spin,
            self.friction_spin,
            self.position_spin,
            self.position_slider,
            self.state_spin,
            self.mimic_enable,
            self.mimic_source_combo,
            self.mimic_auto_check,
            self.mimic_reverse_check,
            self.mimic_multiplier_spin,
            self.mimic_offset_spin,
            self.drive_enable,
            self.drive_source_combo,
            self.drive_max_rpm_spin,
            self.drive_deadband_spin,
            self.drive_reverse_check,
        )
        self._loading = True
        try:
            with _blocked(*widgets):
                self.name_edit.setText(str(getattr(joint, "name", "joint")))
                self.type_combo.setCurrentText(str(getattr(joint, "type", "fixed")))
                self.parent_combo.setCurrentText(str(getattr(joint, "parent", "")))
                self.child_combo.setCurrentText(str(getattr(joint, "child", "")))
                self._refresh_mimic_sources()
                self._refresh_drive_sources()
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
                self.effort_spin.setValue(float(getattr(joint, "effort", 100.0)))
                self.velocity_spin.setValue(
                    float(getattr(joint, "velocity", 1.0)) * scale
                )
                self.damping_spin.setValue(float(getattr(joint, "damping", 0.0)))
                self.friction_spin.setValue(float(getattr(joint, "friction", 0.0)))
                # Refresh the range before loading the position. Otherwise a
                # value can be silently clamped by the previously shown
                # joint's narrower range when switching joint types.
                self._refresh_units()
                self.position_spin.setValue(
                    float(getattr(joint, "position", 0.0)) * scale
                )
                self._sync_slider_from_spin()
                mimic_joint = getattr(joint, "mimic_joint", None)
                self.mimic_enable.setChecked(bool(mimic_joint))
                if mimic_joint:
                    self.mimic_source_combo.setCurrentText(str(mimic_joint))
                self.mimic_auto_check.setChecked(
                    bool(getattr(joint, "mimic_auto", False))
                    if mimic_joint
                    else True
                )
                self.mimic_reverse_check.setChecked(
                    bool(getattr(joint, "mimic_reverse", False))
                )
                self.mimic_multiplier_spin.setValue(
                    float(getattr(joint, "mimic_multiplier", 1.0))
                )
                self.mimic_offset_spin.setValue(
                    float(getattr(joint, "mimic_offset", 0.0)) * scale
                )
                drive_source = getattr(joint, "drive_source_joint", None)
                self.drive_enable.setChecked(bool(drive_source))
                if drive_source:
                    self.drive_source_combo.setCurrentText(str(drive_source))
                max_velocity = float(
                    getattr(joint, "drive_max_velocity", 2.0 * math.pi)
                )
                self.drive_max_rpm_spin.setValue(
                    max_velocity * 60.0 / (2.0 * math.pi)
                )
                self.drive_deadband_spin.setValue(
                    float(getattr(joint, "drive_deadband", 0.03)) * 100.0
                )
                self.drive_reverse_check.setChecked(
                    bool(getattr(joint, "drive_reverse", False))
                )
        finally:
            self._loading = False
        self._refresh_mimic_controls()
        self._refresh_drive_controls()

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
        mimic_source = (
            self.mimic_source_combo.currentData()
            if self.mimic_enable.isChecked()
            else None
        )
        if self.mimic_enable.isChecked() and not mimic_source:
            raise ValueError("연동할 구동 관절을 선택하세요.")
        drive_source = (
            self.drive_source_combo.currentData()
            if self.drive_enable.isChecked() and joint_type == "continuous"
            else None
        )
        if self.drive_enable.isChecked() and joint_type != "continuous":
            raise ValueError("레버 속도 구동 바퀴는 동작 종류가 continuous여야 합니다.")
        if self.drive_enable.isChecked() and not drive_source:
            raise ValueError("전·후진 레버 관절을 선택하세요.")
        if mimic_source and drive_source:
            raise ValueError("mimic 위치 연동과 레버 속도 구동을 동시에 사용할 수 없습니다.")
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
            "effort": self.effort_spin.value(),
            "velocity": self.velocity_spin.value() / scale,
            "damping": self.damping_spin.value(),
            "friction": self.friction_spin.value(),
            "mimic_joint": mimic_source,
            "mimic_auto": bool(
                self.mimic_enable.isChecked()
                and self.mimic_auto_check.isChecked()
            ),
            "mimic_reverse": bool(
                self.mimic_enable.isChecked()
                and self.mimic_reverse_check.isChecked()
            ),
            "mimic_multiplier": self.mimic_multiplier_spin.value(),
            "mimic_offset": self.mimic_offset_spin.value() / scale,
            "drive_source_joint": drive_source,
            "drive_max_velocity": (
                self.drive_max_rpm_spin.value() * 2.0 * math.pi / 60.0
            ),
            "drive_deadband": self.drive_deadband_spin.value() / 100.0,
            "drive_reverse": bool(
                self.drive_enable.isChecked()
                and self.drive_reverse_check.isChecked()
            ),
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
        scalar = kind in {"prismatic", "revolute", "continuous"}
        units = "mm" if kind == "prismatic" else "deg"
        if kind in {"fixed", "planar", "floating"}:
            units = "—"
        self.limit_units.setText(units)
        self.velocity_units.setText(
            "mm/s" if kind == "prismatic" else ("deg/s" if scalar else "—")
        )
        self.position_units.setText(units)
        self.step_units.setText(units)
        self.mimic_offset_units.setText(units)
        self.preview_box.setEnabled(kind in {"prismatic", "revolute", "continuous"})
        self.flip_axis_button.setEnabled(scalar)
        self._refresh_position_range()
        self._refresh_mimic_controls()
        self._refresh_drive_controls()

    @staticmethod
    def _mimic_display_units(joint_type: str) -> tuple[float, str]:
        if joint_type == "prismatic":
            return 1000.0, "mm"
        return 180.0 / math.pi, "°"

    def _refresh_mimic_controls(self, *_args: Any) -> None:
        scalar = self.type_combo.currentText() in {
            "prismatic",
            "revolute",
            "continuous",
        }
        enabled = bool(self.mimic_enable.isChecked() and scalar)
        auto = bool(enabled and self.mimic_auto_check.isChecked())
        self.mimic_enable.setEnabled(scalar)
        self.mimic_source_combo.setEnabled(enabled)
        self.mimic_auto_check.setEnabled(enabled)
        self.mimic_reverse_check.setEnabled(enabled and auto)
        self.mimic_multiplier_spin.setEnabled(enabled and not auto)
        self.mimic_offset_spin.setEnabled(enabled and not auto)
        self._refresh_preview_enabled()

        if not enabled:
            self.mimic_formula_label.setText(
                "독립 관절입니다. 연동을 켜면 구동 관절의 슬라이더를 따라 움직입니다."
            )
            return
        source_name = self.mimic_source_combo.currentData()
        source = self._joint_specs.get(str(source_name)) if source_name else None
        if source is None:
            self.mimic_formula_label.setText("구동 관절을 선택하세요.")
            return
        if not auto:
            self.mimic_formula_label.setText(
                f"{source_name} × {self.mimic_multiplier_spin.value():.6g} "
                f"+ {self.mimic_offset_spin.value():.3f} {self.mimic_offset_units.text()}"
            )
            return

        source_type = str(getattr(source, "type", "fixed"))
        source_scale, source_units = self._mimic_display_units(source_type)
        source_lower = getattr(source, "lower", 0.0)
        source_upper = getattr(source, "upper", 0.0)
        if source_lower is None:
            source_lower = -math.pi if source_type == "continuous" else 0.0
        if source_upper is None:
            source_upper = math.pi if source_type == "continuous" else 0.0
        target_0 = self.lower_spin.value()
        target_1 = self.upper_spin.value()
        if self.mimic_reverse_check.isChecked():
            target_0, target_1 = target_1, target_0
        self.mimic_formula_label.setText(
            f"자동 대응: {source_name} 상태 0 "
            f"({float(source_lower) * source_scale:.3f}{source_units}) → "
            f"{target_0:.3f}{self.mimic_offset_units.text()}\n"
            f"상태 1 ({float(source_upper) * source_scale:.3f}{source_units}) → "
            f"{target_1:.3f}{self.mimic_offset_units.text()}"
        )

    def _refresh_preview_enabled(self) -> None:
        scalar = self.type_combo.currentText() in {
            "prismatic",
            "revolute",
            "continuous",
        }
        coupled = bool(
            self.mimic_enable.isChecked() or self.drive_enable.isChecked()
        )
        self.preview_box.setEnabled(scalar and not coupled)

    def _refresh_drive_controls(self, *_args: Any) -> None:
        continuous = self.type_combo.currentText() == "continuous"
        enabled = bool(self.drive_enable.isChecked() and continuous)
        self.drive_enable.setEnabled(continuous)
        self.drive_source_combo.setEnabled(enabled)
        self.drive_max_rpm_spin.setEnabled(enabled)
        self.drive_deadband_spin.setEnabled(enabled)
        self.drive_reverse_check.setEnabled(enabled)
        self._refresh_preview_enabled()

        if not continuous:
            self.drive_formula_label.setText(
                "바퀴 관절의 동작 종류를 continuous로 설정해야 합니다."
            )
            return
        if not enabled:
            self.drive_formula_label.setText(
                "속도 구동을 켜면 레버 중앙에서 정지하고 양 끝으로 갈수록 빨라집니다."
            )
            return
        source_name = self.drive_source_combo.currentData()
        source = self._joint_specs.get(str(source_name)) if source_name else None
        if source is None:
            self.drive_formula_label.setText("전·후진 레버 관절을 선택하세요.")
            return
        source_type = str(getattr(source, "type", "revolute"))
        source_scale, source_units = self._mimic_display_units(source_type)
        lower = float(getattr(source, "lower", 0.0) or 0.0) * source_scale
        upper = float(getattr(source, "upper", 0.0) or 0.0) * source_scale
        middle = (lower + upper) * 0.5
        rpm = self.drive_max_rpm_spin.value()
        reverse = -1.0 if self.drive_reverse_check.isChecked() else 1.0
        reverse_rpm = -rpm * reverse
        forward_rpm = rpm * reverse
        self.drive_formula_label.setText(
            f"상태 0 ({lower:.3f}{source_units}) → {reverse_rpm:+.1f} RPM\n"
            f"중앙 ({middle:.3f}{source_units}) → 정지\n"
            f"상태 1 ({upper:.3f}{source_units}) → {forward_rpm:+.1f} RPM\n"
            "레버 위치를 정한 뒤 Play를 누르면 계속 회전합니다."
        )

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
