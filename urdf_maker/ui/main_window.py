from __future__ import annotations

import colorsys
import copy
import hashlib
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSettings, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..collision import MeshCollisionChecker
from ..model import (
    JointSpec,
    ProjectValidationError,
    RobotProject,
    apply_transform,
    axis_angle_matrix,
    rpy_matrix,
    sanitize_name,
)
from ..project_io import apply_project_config, load_project_config, save_project
from ..step_subprocess import load_step_project_isolated
from ..urdf_io import export_urdf, load_urdf
from .editors import JointEditorWidget, NewLinkDialog
from .viewport import ViewportWidget
from .wizards import MechanismWizard, SimulationExportDialog


ROLE_ID = Qt.ItemDataRole.UserRole
ROLE_JOINT = Qt.ItemDataRole.UserRole + 1
ROLE_COLLISION_PARTS = Qt.ItemDataRole.UserRole + 2
COLLISION_DISPLAY_RGB = (0.96, 0.08, 0.04)


@dataclass(frozen=True)
class _CollisionIssue:
    message: str
    part_ids: tuple[str, ...] = ()
    is_collision: bool = True


def _stable_part_display_color(
    part_id: str,
    alpha: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return a deterministic display color tied only to a stable part id."""

    digest = hashlib.blake2s(str(part_id).encode("utf-8"), digest_size=4).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    saturation = 0.52 + digest[2] / 255.0 * 0.18
    value = 0.82 + digest[3] / 255.0 * 0.12
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return red, green, blue, max(0.0, min(1.0, float(alpha)))


def _geometry_principal_axes(vertices: Iterable[np.ndarray]) -> dict[str, np.ndarray]:
    """Return deterministic long/middle/thickness axes for selected geometry."""

    arrays = [
        np.asarray(value, dtype=float).reshape((-1, 3))
        for value in vertices
        if np.asarray(value).size
    ]
    if not arrays:
        return {}
    points = np.vstack(arrays)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        return {}
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / float(len(centered))
    try:
        _values, vectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return {}
    vectors = vectors[:, ::-1]
    for index in range(3):
        vector = vectors[:, index]
        dominant = int(np.argmax(np.abs(vector)))
        if vector[dominant] < 0.0:
            vectors[:, index] *= -1.0
    return {
        name: vectors[:, index].copy()
        for index, name in enumerate(("A", "B", "C"))
    }


def _coerce_feature_axis(raw: Any) -> dict[str, Any] | None:
    try:
        origin = np.asarray(raw["origin"], dtype=float)
        direction = np.asarray(raw["direction"], dtype=float)
        radius = float(raw.get("radius", 0.0))
        length = float(raw.get("length", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    norm = float(np.linalg.norm(direction))
    if (
        origin.shape != (3,)
        or direction.shape != (3,)
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(direction))
        or norm <= 1.0e-12
        or not math.isfinite(radius)
    ):
        return None
    return {
        "origin": origin,
        "direction": direction / norm,
        "radius": max(radius, 0.0),
        "length": max(length, 0.0) if math.isfinite(length) else 0.0,
    }


def _axis_line_distance(
    first_origin: np.ndarray,
    first_direction: np.ndarray,
    second_origin: np.ndarray,
) -> float:
    delta = second_origin - first_origin
    return float(
        np.linalg.norm(delta - np.dot(delta, first_direction) * first_direction)
    )


def _cad_joint_axis_candidates(
    child_parts: Iterable[Any],
    parent_parts: Iterable[Any],
    *,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Rank exact STEP cylinder axes shared by child and parent geometry."""

    child_axes = [
        candidate
        for part in child_parts
        for raw in getattr(part, "feature_axes", ())
        if (candidate := _coerce_feature_axis(raw)) is not None
    ]
    parent_axes = [
        candidate
        for part in parent_parts
        for raw in getattr(part, "feature_axes", ())
        if (candidate := _coerce_feature_axis(raw)) is not None
    ]
    ranked: list[dict[str, Any]] = []
    for parent_axis in parent_axes:
        for child_axis in child_axes:
            alignment = abs(
                float(
                    np.dot(
                        parent_axis["direction"],
                        child_axis["direction"],
                    )
                )
            )
            if alignment < 0.9995:
                continue
            distance = _axis_line_distance(
                parent_axis["origin"],
                parent_axis["direction"],
                child_axis["origin"],
            )
            radius_scale = max(
                parent_axis["radius"],
                child_axis["radius"],
                0.001,
            )
            # STEP cylinder axes are analytic, so a generous multi-millimetre
            # allowance tends to join merely adjacent holes.  Keep a small
            # manufacturing/authoring tolerance while requiring the same
            # practical centerline.  Radius is deliberately not a hard gate:
            # shafts and bores commonly have different fit diameters.
            distance_limit = max(0.0002, min(radius_scale * 0.02, 0.001))
            if distance > distance_limit:
                continue
            radius_difference = (
                abs(parent_axis["radius"] - child_axis["radius"])
                / radius_scale
            )
            score = (
                distance / distance_limit
                + radius_difference * 0.75
                + (1.0 - alignment) * 20.0
            )
            ranked.append(
                {
                    "origin": parent_axis["origin"].copy(),
                    "direction": parent_axis["direction"].copy(),
                    "parent_radius": parent_axis["radius"],
                    "child_radius": child_axis["radius"],
                    "score": score,
                    "shared": True,
                }
            )

    # A cylinder found only on the selected child is often a wheel, bearing,
    # or decorative hole and says nothing about how the child moves relative
    # to its parent.  Do not let those axes occupy all A/B/C slots.  Without a
    # matching parent centerline the caller falls back to the selected bundle's
    # principal BBox directions, whose A axis is the useful long/horizontal
    # translation direction for racks and axle bars.

    unique: list[dict[str, Any]] = []
    for candidate in sorted(ranked, key=lambda item: item["score"]):
        duplicate = False
        for existing in unique:
            alignment = abs(
                float(np.dot(candidate["direction"], existing["direction"]))
            )
            if alignment < 0.999:
                continue
            if (
                _axis_line_distance(
                    existing["origin"],
                    existing["direction"],
                    candidate["origin"],
                )
                <= 0.001
            ):
                duplicate = True
                break
        if duplicate:
            continue
        unique.append(candidate)
        if len(unique) >= limit:
            break
    return unique


class _WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class _FunctionWorker(QRunnable):
    def __init__(self, function: Callable[[], Any]) -> None:
        super().__init__()
        self.function = function
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function()
        except Exception:
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.finished.emit(result)


class MainWindow(QMainWindow):
    """Desktop editor joining STEP selection, URDF topology and live FK preview."""

    def __init__(self) -> None:
        super().__init__()
        self.project: RobotProject | None = None
        self.project_path: Path | None = None
        self.current_link: str | None = None
        self.current_joint: str | None = None
        self._dirty = False
        self._selection_guard = False
        self._tree_guard = False
        self._link_parts_guard = False
        self._hide_assigned_parts = False
        self._scene_scale = 0.2
        self._workers: set[_FunctionWorker] = set()
        self._progress: QProgressDialog | None = None
        self._settings = QSettings()
        self._demo_original_positions: dict[str, float] = {}
        self._demo_joints: list[JointSpec] = []
        self._demo_drive_joints: list[JointSpec] = []
        self._animation_mode: str | None = None
        self._operator_panel_visibility: tuple[bool, bool, bool] | None = None
        self._demo_started_at = 0.0
        self._demo_last_tick = 0.0
        self._collision_precheck_count = 0
        self._collision_precheck_issues_cache: list[_CollisionIssue] = []
        self._collision_highlight_part_ids: set[str] = set()
        self._collision_checker: MeshCollisionChecker | None = None
        self._collision_check_pending = False
        self._collision_worker_active = False
        self._collision_rescan_requested = False
        self._new_link_dialog: NewLinkDialog | None = None
        self._pending_mechanism_preset: dict[str, Any] | None = None
        self._demo_timer = QTimer(self)
        self._demo_timer.setInterval(33)
        self._demo_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._demo_timer.timeout.connect(self._advance_demo_animation)

        self.setWindowTitle("STEP URDF Maker")
        self.resize(1540, 920)
        self.setMinimumSize(1080, 680)
        # Keep the inspector as one attached panel. In particular, do not let
        # Qt turn the right and bottom docks into tabbed/floating duplicates
        # while the native VTK surface is recreated during maximize/restore.
        self.setDockOptions(QMainWindow.DockOption.AnimatedDocks)
        self.setCorner(
            Qt.Corner.BottomRightCorner,
            Qt.DockWidgetArea.BottomDockWidgetArea,
        )
        self.setAcceptDrops(True)
        self._build_actions()
        self._build_ui()
        self._apply_style()
        self._set_actions_enabled(False)
        self.statusBar().showMessage("STEP 또는 URDF를 불러오세요.")

    # ---------- UI construction ----------
    def _build_actions(self) -> None:
        self.open_step_action = QAction("STEP 열기…", self)
        self.open_step_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_step_action.triggered.connect(self.open_step_dialog)
        self.open_urdf_action = QAction("URDF 열기…", self)
        self.open_urdf_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.open_urdf_action.triggered.connect(self.open_urdf_dialog)
        self.open_project_action = QAction("프로젝트 열기…", self)
        self.open_project_action.triggered.connect(self.open_project_dialog)
        self.save_project_action = QAction("프로젝트 저장", self)
        self.save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_project_action.triggered.connect(self.save_project)
        self.save_as_action = QAction("프로젝트 다른 이름으로 저장…", self)
        self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.save_as_action.triggered.connect(lambda: self.save_project(save_as=True))
        self.export_action = QAction("URDF 패키지 내보내기…", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.triggered.connect(self.export_project)
        self.frame_action = QAction("전체 보기", self)
        self.frame_action.setShortcut(QKeySequence("F"))
        self.frame_action.triggered.connect(lambda: self.viewport.frame_all())
        self.clear_selection_action = QAction("선택 해제", self)
        self.clear_selection_action.setShortcut(QKeySequence.StandardKey.Cancel)
        self.clear_selection_action.triggered.connect(lambda: self._set_selected_parts([]))
        self.mechanism_wizard_action = QAction("대표 기구 마법사…", self)
        self.mechanism_wizard_action.setShortcut(QKeySequence("Ctrl+M"))
        self.mechanism_wizard_action.setToolTip(
            "문·슬라이더·회전체·연동 기구·컨베이어의 기본 관절을 단계별로 만듭니다."
        )
        self.mechanism_wizard_action.triggered.connect(self.open_mechanism_wizard)

        file_menu = self.menuBar().addMenu("파일")
        file_menu.addAction(self.open_step_action)
        file_menu.addAction(self.open_urdf_action)
        file_menu.addAction(self.open_project_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction("종료", self.close)

        view_menu = self.menuBar().addMenu("보기")
        view_menu.addAction(self.frame_action)
        view_menu.addAction(self.clear_selection_action)
        self.part_colors_action = QAction("파트 구분 색상", self)
        self.part_colors_action.setCheckable(True)
        self.part_colors_action.setChecked(True)
        self.part_colors_action.setToolTip("원본 재질을 바꾸지 않고 화면에서만 파츠별 색을 구분합니다.")
        self.part_colors_action.triggered.connect(
            lambda: self._rebuild_viewport(self._selected_part_ids())
        )
        view_menu.addAction(self.part_colors_action)
        self.hide_assigned_action = QAction("사용한 파츠 숨기기", self)
        self.hide_assigned_action.setCheckable(True)
        self.hide_assigned_action.setShortcut(QKeySequence("Ctrl+H"))
        self.hide_assigned_action.setToolTip(
            "링크에 이미 배정된 파츠를 목록과 3D에서 임시로 숨깁니다."
        )
        self.hide_assigned_action.toggled.connect(self._assigned_visibility_changed)
        view_menu.addAction(self.hide_assigned_action)

        configure_menu = self.menuBar().addMenu("구성")
        configure_menu.addAction(self.mechanism_wizard_action)

        toolbar = self.addToolBar("기본 도구")
        toolbar.setMovable(False)
        toolbar.addAction(self.open_step_action)
        toolbar.addAction(self.open_urdf_action)
        toolbar.addSeparator()
        toolbar.addAction(self.save_project_action)
        toolbar.addAction(self.export_action)
        toolbar.addSeparator()
        toolbar.addAction(self.frame_action)
        toolbar.addSeparator()
        toolbar.addAction(self.mechanism_wizard_action)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        self.left_panel = left
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 4, 8)
        title = QLabel("형상과 링크")
        title.setObjectName("panelTitle")
        left_layout.addWidget(title)
        self.left_tabs = QTabWidget()
        left_layout.addWidget(self.left_tabs, 1)

        part_page = QWidget()
        part_layout = QVBoxLayout(part_page)
        part_layout.setContentsMargins(4, 6, 4, 4)
        part_hint = QLabel(
            "목록 또는 3D에서 움직일 형상을 고릅니다. "
            "파트별 체크: 체크=보임, 체크 해제=숨김. "
            "3D: Ctrl=추가, Shift/Alt=선택에서 빼기."
        )
        part_hint.setWordWrap(True)
        part_layout.addWidget(part_hint)
        self.part_filter_label = QLabel()
        self.part_filter_label.setWordWrap(True)
        part_layout.addWidget(self.part_filter_label)
        self.part_tree = QTreeWidget()
        self.part_tree.setHeaderLabels(["형상", "현재 링크"])
        self.part_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.part_tree.setAlternatingRowColors(True)
        self.part_tree.setUniformRowHeights(True)
        self.part_tree.itemSelectionChanged.connect(self._part_tree_selection_changed)
        self.part_tree.itemChanged.connect(self._part_tree_item_changed)
        part_layout.addWidget(self.part_tree, 1)
        selection_buttons = QHBoxLayout()
        self.create_link_button = QPushButton("+ 선택 형상으로 자식 링크")
        self.create_link_button.clicked.connect(self.create_link_from_selection)
        self.assign_button = QPushButton("선택 → 현재 링크")
        self.assign_button.clicked.connect(self.assign_selection_to_current_link)
        selection_buttons.addWidget(self.create_link_button, 1)
        selection_buttons.addWidget(self.assign_button)
        part_layout.addLayout(selection_buttons)
        self.to_base_button = QPushButton("선택 → Base(root) 링크")
        self.to_base_button.clicked.connect(self.assign_selection_to_base)
        part_layout.addWidget(self.to_base_button)
        self.assigned_visibility_check = QCheckBox(
            "링크에 배정된 파츠 숨기기 (Ctrl+H)"
        )
        self.assigned_visibility_check.setToolTip(
            "체크하면 이미 링크에 배정된 파츠를 형상 목록과 3D에서 임시로 숨깁니다."
        )
        self.assigned_visibility_check.toggled.connect(
            self.hide_assigned_action.setChecked
        )
        part_layout.addWidget(self.assigned_visibility_check)
        self._update_assigned_visibility_controls()
        self.left_tabs.addTab(part_page, "형상 선택")

        link_page = QWidget()
        link_layout = QVBoxLayout(link_page)
        link_layout.setContentsMargins(4, 6, 4, 4)
        link_hint = QLabel(
            "링크를 선택하면 아래에 그 링크의 형상이 표시됩니다. "
            "3D/형상 목록에서 고른 파츠를 추가하거나, 아래 목록에서 골라 뺄 수 있습니다."
        )
        link_hint.setWordWrap(True)
        link_layout.addWidget(link_hint)
        tree_buttons = QHBoxLayout()
        self.new_tree_button = QPushButton("전체 새로 시작")
        self.new_tree_button.setToolTip("기존 링크/관절을 지우고 새 Base 링크에서 시작합니다.")
        self.new_tree_button.clicked.connect(self.new_manual_tree)
        self.add_child_button = QPushButton("+ 선택 형상으로 자식 링크")
        self.add_child_button.setToolTip(
            "먼저 움직일 형상을 선택한 뒤 부모 링크 위에서 관절 중심과 축을 설정합니다."
        )
        self.add_child_button.clicked.connect(self.add_child_link_from_selection)
        tree_buttons.addWidget(self.new_tree_button)
        tree_buttons.addWidget(self.add_child_button, 1)
        link_layout.addLayout(tree_buttons)
        self.mechanism_wizard_button = QPushButton("대표 기구 마법사…  (Ctrl+M)")
        self.mechanism_wizard_button.setToolTip(
            "선택한 형상으로 문·뚜껑, 슬라이더, 회전체, mimic 연동 또는 "
            "컨베이어 롤러를 만든 뒤 3D에서 축을 정밀 조정합니다."
        )
        self.mechanism_wizard_button.clicked.connect(self.open_mechanism_wizard)
        link_layout.addWidget(self.mechanism_wizard_button)
        self.link_tree = QTreeWidget()
        self.link_tree.setHeaderLabels(["링크 (형상 수)", "0→1 동작"])
        self.link_tree.setAlternatingRowColors(True)
        self.link_tree.setUniformRowHeights(True)
        self.link_tree.itemSelectionChanged.connect(self._link_tree_selection_changed)
        link_layout.addWidget(self.link_tree, 2)
        self.link_parts_label = QLabel("현재 링크 형상 (0개)")
        link_layout.addWidget(self.link_parts_label)
        self.link_parts_list = QListWidget()
        self.link_parts_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.link_parts_list.setMinimumHeight(90)
        self.link_parts_list.setToolTip(
            "현재 링크에서 뺄 형상을 하나 이상 선택하세요. Ctrl로 여러 개를 선택할 수 있습니다."
        )
        self.link_parts_list.itemSelectionChanged.connect(
            self._link_parts_selection_changed
        )
        link_layout.addWidget(self.link_parts_list, 1)
        membership_buttons = QHBoxLayout()
        self.assign_current_tree_button = QPushButton("+ 3D/형상 선택 추가")
        self.assign_current_tree_button.setToolTip(
            "3D 화면 또는 형상 선택 탭에서 선택한 파츠를 현재 링크로 이동합니다."
        )
        self.assign_current_tree_button.clicked.connect(self.assign_selection_to_current_link)
        self.unassign_current_tree_button = QPushButton("− 목록 선택 빼기")
        self.unassign_current_tree_button.setToolTip(
            "위의 현재 링크 형상 목록에서 선택한 파츠를 미할당 상태로 뺍니다."
        )
        self.unassign_current_tree_button.clicked.connect(
            self.unassign_selection_from_current_link
        )
        membership_buttons.addWidget(self.assign_current_tree_button, 1)
        membership_buttons.addWidget(self.unassign_current_tree_button, 1)
        link_layout.addLayout(membership_buttons)
        self.merge_button = QPushButton("선택 링크 삭제")
        self.merge_button.setToolTip(
            "파트는 미할당 상태가 되고, 하위 링크는 부모 아래에 유지됩니다."
        )
        self.merge_button.clicked.connect(self.delete_current_link)
        link_layout.addWidget(self.merge_button)
        self.left_tabs.addTab(link_page, "URDF 링크 트리")

        splitter.addWidget(left)

        self.viewport = ViewportWidget()
        self.viewport.partsSelectionChanged.connect(self._viewport_selection_changed)
        self.viewport.animationToggled.connect(self._set_demo_playing)
        self.viewport.controlAnimationToggled.connect(self._set_control_playing)
        self.viewport.operatorControlChanged.connect(
            self._set_operator_control_value
        )
        splitter.addWidget(self.viewport)

        right_contents = QWidget()
        right_contents.setSizePolicy(
            QSizePolicy.Policy.Ignored,
            QSizePolicy.Policy.Preferred,
        )
        right_layout = QVBoxLayout(right_contents)
        right_layout.setContentsMargins(8, 8, 8, 12)
        self.selection_label = QLabel("선택된 형상 없음")
        self.selection_label.setWordWrap(True)
        self.selection_label.setObjectName("selectionSummary")
        right_layout.addWidget(self.selection_label)
        self.joint_editor = JointEditorWidget()
        self.joint_editor.applyRequested.connect(self.apply_joint_values)
        self.joint_editor.positionChanged.connect(self.set_current_joint_position)
        self.joint_editor.originFromSelectionRequested.connect(self.use_selection_center_for_origin)
        self.joint_editor.axisPreviewRequested.connect(self.preview_joint_axis)
        right_layout.addWidget(self.joint_editor)
        right_layout.addStretch(1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setWidget(right_contents)
        right_scroll.setMinimumWidth(280)
        right_scroll.setObjectName("jointInspectorScroll")

        self.inspector_dock = QDockWidget("구성 및 동작 시험", self)
        self.inspector_dock.setObjectName("jointInspectorDock")
        self.inspector_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.inspector_dock.setFeatures(
            QDockWidget.DockWidgetFeature.NoDockWidgetFeatures
        )
        self.inspector_dock.setMinimumWidth(300)
        self.inspector_dock.setWidget(right_scroll)

        splitter.setSizes([300, 900])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea,
            self.inspector_dock,
        )
        self.resizeDocks(
            [self.inspector_dock],
            [360],
            Qt.Orientation.Horizontal,
        )

        issues_dock = QDockWidget("검증 및 알림", self)
        self.issues_dock = issues_dock
        issues_dock.setObjectName("issuesDock")
        issues_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.issue_list = QListWidget()
        self.issue_list.setMaximumHeight(130)
        self.issue_list.setToolTip(
            "충돌 항목을 선택하면 관련 파트를 3D 화면에서 빨간색으로 강조합니다."
        )
        self.issue_list.currentItemChanged.connect(
            self._issue_selection_changed
        )
        issues_dock.setWidget(self.issue_list)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, issues_dock)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: palette(window); }
            QLabel#panelTitle { font-size: 15px; font-weight: 600; padding: 4px 2px 8px 2px; }
            QLabel#selectionSummary { padding: 8px; border: 1px solid palette(mid); border-radius: 5px; }
            QGroupBox { font-weight: 600; margin-top: 10px; padding-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QPushButton { min-height: 26px; padding: 3px 8px; }
            QTreeWidget { border: 1px solid palette(mid); }
            """
        )

    def _assigned_visibility_changed(self, checked: bool) -> None:
        self._hide_assigned_parts = bool(checked)
        self._update_assigned_visibility_controls()
        if self.project is None:
            return
        selected = [
            part_id
            for part_id in self._selected_part_ids()
            if (
                part_id in self.project.parts
                and not (
                    self._hide_assigned_parts
                    and self.project.parts[part_id].link_name is not None
                )
            )
        ]
        self._rebuild_part_tree()
        self._rebuild_viewport(selected)
        self._set_selected_parts(selected)

    def _update_assigned_visibility_controls(self) -> None:
        hidden = bool(self._hide_assigned_parts)
        if hasattr(self, "hide_assigned_action"):
            self.hide_assigned_action.setText("링크에 배정된 파츠 숨기기")
        if hasattr(self, "assigned_visibility_check"):
            self.assigned_visibility_check.blockSignals(True)
            self.assigned_visibility_check.setChecked(hidden)
            self.assigned_visibility_check.blockSignals(False)
        if hasattr(self, "part_filter_label"):
            assigned_count = (
                sum(
                    part.link_name is not None
                    for part in self.project.parts.values()
                )
                if self.project is not None
                else 0
            )
            if hidden:
                self.part_filter_label.setText(
                    f"미할당 파츠만 표시 중 · 사용한 파츠 {assigned_count}개 숨김"
                )
            else:
                self.part_filter_label.setText(
                    f"전체 파츠 표시 중 · 사용한 파츠 {assigned_count}개"
                )

    # ---------- file operations ----------
    def _last_directory(self) -> str:
        return str(self._settings.value("lastDirectory", str(Path.cwd())))

    def _remember_directory(self, path: str | Path) -> None:
        self._settings.setValue("lastDirectory", str(Path(path).resolve().parent))

    def open_step_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "STEP 파일 열기", self._last_directory(), "STEP (*.step *.stp);;모든 파일 (*)"
        )
        if path:
            self.open_path(Path(path))

    def open_urdf_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "URDF 파일 열기", self._last_directory(), "URDF (*.urdf *.xml);;모든 파일 (*)"
        )
        if path:
            self.open_path(Path(path))

    def open_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "URDF Maker 프로젝트 열기",
            self._last_directory(),
            "URDF Maker 프로젝트 (*.urdfmaker.json *.json)",
        )
        if path:
            self.open_path(Path(path))

    def open_path(self, path: str | Path) -> None:
        source = Path(path).expanduser().resolve()
        if not source.exists():
            QMessageBox.critical(self, "파일 없음", str(source))
            return
        if not self._confirm_discard_changes():
            return
        self._remember_directory(source)
        suffix = source.suffix.lower()
        if suffix in {".step", ".stp"}:
            self._load_step_path(source)
        elif suffix in {".urdf", ".xml"}:
            self._load_urdf_path(source)
        elif suffix == ".json":
            self._load_project_path(source)
        else:
            QMessageBox.warning(self, "지원하지 않는 파일", f"지원하지 않는 확장자입니다: {suffix}")

    def _load_step_path(
        self,
        path: Path,
        after: Callable[[RobotProject], None] | None = None,
    ) -> None:
        def operation() -> RobotProject:
            # OCCT is native code.  Keep it outside the Qt/VTK process so a
            # DLL-level import failure becomes a reportable worker error rather
            # than closing the complete desktop application.
            project = load_step_project_isolated(path, linear_deflection=0.001)
            if not project.links:
                project.create_link("base_link", project.parts.keys())
                project.root_link = "base_link"
            return project

        self._run_background(
            f"STEP 불러오는 중…\n{path.name}",
            operation,
            after or (lambda project: self._set_project(project, None)),
        )

    def _load_urdf_path(
        self,
        path: Path,
        after: Callable[[RobotProject], None] | None = None,
    ) -> None:
        self._run_background(
            f"URDF 불러오는 중…\n{path.name}",
            lambda: load_urdf(path),
            after or (lambda project: self._set_project(project, None)),
        )

    def _load_project_path(self, project_path: Path) -> None:
        try:
            payload, source_path, source_kind = load_project_config(project_path)
        except Exception as exc:
            QMessageBox.critical(self, "프로젝트 열기 실패", str(exc))
            return
        if source_path is None or not source_path.exists():
            replacement, _ = QFileDialog.getOpenFileName(
                self,
                "프로젝트의 원본 STEP/URDF 위치 지정",
                self._last_directory(),
                "지원 파일 (*.step *.stp *.urdf *.xml)",
            )
            if not replacement:
                return
            source_path = Path(replacement).resolve()
        kind = (source_kind or source_path.suffix.lstrip(".")).lower()

        def finish(base_project: RobotProject) -> None:
            warnings = apply_project_config(base_project, payload)
            self._set_project(base_project, project_path)
            self._add_issues(warnings, prefix="프로젝트: ")

        if kind in {"step", "stp"}:
            self._load_step_path(source_path, finish)
        else:
            self._load_urdf_path(source_path, finish)

    def _run_background(
        self,
        label: str,
        operation: Callable[[], Any],
        completed: Callable[[Any], None],
    ) -> None:
        if self._progress is not None:
            return
        dialog = QProgressDialog(label, "", 0, 0, self)
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.show()
        self._progress = dialog
        worker = _FunctionWorker(operation)
        self._workers.add(worker)

        def cleanup() -> None:
            self._workers.discard(worker)
            if self._progress is dialog:
                self._progress.close()
                self._progress = None

        def success(result: Any) -> None:
            cleanup()
            try:
                completed(result)
            except Exception:
                QMessageBox.critical(self, "불러오기 처리 실패", traceback.format_exc())

        def failure(details: str) -> None:
            cleanup()
            QMessageBox.critical(self, "불러오기 실패", details)

        worker.signals.finished.connect(success)
        worker.signals.failed.connect(failure)
        QThreadPool.globalInstance().start(worker)

    def save_project(self, save_as: bool = False) -> bool:
        if self.project is None:
            return False
        destination = self.project_path
        if save_as or destination is None:
            suggested = f"{sanitize_name(self.project.name, 'robot')}.urdfmaker.json"
            path, _ = QFileDialog.getSaveFileName(
                self,
                "프로젝트 저장",
                str(Path(self._last_directory()) / suggested),
                "URDF Maker 프로젝트 (*.urdfmaker.json)",
            )
            if not path:
                return False
            destination = Path(path)
            if not str(destination).lower().endswith(".urdfmaker.json"):
                destination = Path(str(destination) + ".urdfmaker.json")
        try:
            self.project_path = save_project(self.project, destination)
        except Exception as exc:
            QMessageBox.critical(self, "저장 실패", str(exc))
            return False
        self._dirty = False
        self._update_title()
        self.statusBar().showMessage(f"프로젝트 저장: {self.project_path}", 7000)
        return True

    def export_project(self) -> None:
        if self.project is None:
            return
        unassigned = [
            part.name for part in self.project.parts.values() if part.link_name is None
        ]
        if unassigned:
            QMessageBox.warning(
                self,
                "미할당 형상 있음",
                f"링크에 넣지 않은 형상 {len(unassigned)}개가 있습니다. "
                "내보내기 전에 Base 또는 자식 링크에 배정하세요.",
            )
            self._refresh_issues()
            return
        errors = self.project.validate(check_names=False)
        if errors:
            QMessageBox.warning(self, "내보낼 수 없음", "\n".join(errors))
            self._refresh_issues()
            return
        parent_dir = QFileDialog.getExistingDirectory(
            self, "URDF 패키지를 만들 상위 폴더", self._last_directory()
        )
        if not parent_dir:
            return
        default_name = f"{sanitize_name(self.project.name.lower(), 'robot')}_description"
        package_name, accepted = QInputDialog.getText(
            self, "패키지 이름", "ROS 패키지 이름", text=default_name
        )
        if not accepted or not package_name.strip():
            return
        safe_dir = sanitize_name(package_name.strip().lower(), "robot_description").replace(".", "_")
        output_dir = Path(parent_dir) / safe_dir
        export_options = SimulationExportDialog(self)
        if export_options.exec() != QDialog.DialogCode.Accepted:
            return
        options = export_options.values()
        try:
            urdf_path = export_urdf(
                self.project,
                output_dir,
                package_name=safe_dir,
                include_collision=bool(options["include_collision"]),
                include_inertial=bool(options["include_inertial"]),
                density=float(options["density"]),
            )
        except Exception as exc:
            QMessageBox.critical(self, "URDF 내보내기 실패", str(exc))
            return
        self.statusBar().showMessage(f"URDF 생성: {urdf_path}", 10000)
        QMessageBox.information(self, "내보내기 완료", f"생성된 URDF:\n{urdf_path}")

    # ---------- project and scene ----------
    def _set_project(self, project: RobotProject, project_path: Path | None) -> None:
        if self._new_link_dialog is not None:
            self._new_link_dialog.reject()
        self._pending_mechanism_preset = None
        self._stop_demo_animation(restore=True)
        self.project = project
        self.project_path = project_path
        self.current_link = project.root_link
        self.current_joint = None
        self._hide_assigned_parts = False
        self.hide_assigned_action.blockSignals(True)
        self.hide_assigned_action.setChecked(False)
        self.hide_assigned_action.blockSignals(False)
        self._update_assigned_visibility_controls()
        self._dirty = False
        self._compute_scene_scale()
        self._collision_checker = MeshCollisionChecker(project)
        self._collision_precheck_count = 0
        self._collision_precheck_issues_cache = []
        self._collision_highlight_part_ids = set()
        self._collision_check_pending = False
        self._collision_rescan_requested = False
        self._rebuild_all()
        self.viewport.frame_all()
        self._update_title()
        self._set_actions_enabled(True)
        self.statusBar().showMessage(
            f"{len(project.parts)}개 형상 · {len(project.links)}개 링크 · {len(project.joints)}개 관절",
            10000,
        )
        self._schedule_collision_precheck()

    def _compute_scene_scale(self) -> None:
        if self.project is None:
            self._scene_scale = 0.2
            return
        vertices = [part.vertices_zero for part in self.project.parts.values() if len(part.vertices_zero)]
        if not vertices:
            self._scene_scale = 0.2
            return
        minimum = np.min([value.min(axis=0) for value in vertices], axis=0)
        maximum = np.max([value.max(axis=0) for value in vertices], axis=0)
        self._scene_scale = max(float(np.linalg.norm(maximum - minimum)), 0.01)

    def _rebuild_all(self, *, preserve_selection: bool = True) -> None:
        selected = self._selected_part_ids() if preserve_selection else []
        self._rebuild_part_tree()
        self._rebuild_link_tree()
        self._rebuild_viewport(selected)
        self._refresh_editor()
        self._refresh_issues()
        self._update_assigned_visibility_controls()

    def _rebuild_part_tree(self) -> None:
        self._tree_guard = True
        try:
            self.part_tree.clear()
            if self.project is None:
                return
            occurrences = {
                str(item.get("id")): item
                for item in self.project.metadata.get("occurrences", [])
                if isinstance(item, dict) and item.get("id") is not None
            }
            assembly_items: dict[tuple[str, ...], QTreeWidgetItem] = {}
            for part in self.project.parts.values():
                if self._hide_assigned_parts and part.link_name is not None:
                    continue
                occurrence = occurrences.get(part.id, {})
                raw_path = occurrence.get("assembly_path", [])
                path = [str(value).strip() for value in raw_path if str(value).strip()]
                # The final XCAF path component is normally the leaf occurrence
                # itself; only preceding components become assembly folders.
                assembly_path = path[:-1] if len(path) > 1 else []
                parent_item: QTreeWidgetItem | None = None
                for depth in range(len(assembly_path)):
                    key = tuple(assembly_path[: depth + 1])
                    group_item = assembly_items.get(key)
                    if group_item is None:
                        group_item = QTreeWidgetItem([assembly_path[depth], "어셈블리"])
                        group_item.setData(0, ROLE_ID, None)
                        if parent_item is None:
                            self.part_tree.addTopLevelItem(group_item)
                        else:
                            parent_item.addChild(group_item)
                        assembly_items[key] = group_item
                    parent_item = group_item
                item = QTreeWidgetItem([part.name, part.link_name or "미할당"])
                item.setData(0, ROLE_ID, part.id)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(0, Qt.CheckState.Checked if part.visible else Qt.CheckState.Unchecked)
                if parent_item is None:
                    self.part_tree.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)
            self.part_tree.expandToDepth(1)
            self.part_tree.resizeColumnToContents(0)
        finally:
            self._tree_guard = False

    def _rebuild_link_tree(self) -> None:
        self.link_tree.clear()
        if self.project is None:
            self._rebuild_current_link_parts()
            return
        added: set[str] = set()

        def add_link(link_name: str, parent_item: QTreeWidgetItem | None) -> None:
            if link_name in added or link_name not in self.project.links:
                return
            added.add(link_name)
            incoming = self.project.joint_for_child(link_name)
            part_count = len(self.project.links[link_name].part_ids)
            item = QTreeWidgetItem(
                [
                    f"{link_name} ({part_count})",
                    self._joint_motion_summary(incoming),
                ]
            )
            item.setData(0, ROLE_ID, link_name)
            item.setData(0, ROLE_JOINT, incoming.name if incoming else None)
            if incoming is not None:
                item.setToolTip(1, f"관절 이름: {incoming.name}")
            if parent_item is None:
                self.link_tree.addTopLevelItem(item)
            else:
                parent_item.addChild(item)
            for child_joint in self.project.children_of(link_name):
                add_link(child_joint.child, item)

        if self.project.root_link:
            add_link(self.project.root_link, None)
        for link_name in self.project.links:
            add_link(link_name, None)
        self.link_tree.expandAll()
        self.link_tree.resizeColumnToContents(0)
        self._rebuild_current_link_parts()

    @staticmethod
    def _joint_motion_summary(joint: JointSpec | None) -> str:
        if joint is None:
            return "BASE · 기준 링크"
        if joint.type == "fixed":
            return "고정"

        axis = np.asarray(joint.axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if norm > 1.0e-12:
            axis = axis / norm
        shortcuts = {
            (1.0, 0.0, 0.0): "+X",
            (-1.0, 0.0, 0.0): "−X",
            (0.0, 1.0, 0.0): "+Y",
            (0.0, -1.0, 0.0): "−Y",
            (0.0, 0.0, 1.0): "+Z",
            (0.0, 0.0, -1.0): "−Z",
        }
        axis_key = tuple(float(round(value, 6)) for value in axis)
        axis_text = shortcuts.get(
            axis_key,
            "(" + ", ".join(f"{value:.2f}" for value in axis) + ")",
        )
        mimic_text = ""
        if joint.mimic_joint:
            mode = "상태 연동" if joint.mimic_auto else f"×{joint.mimic_multiplier:g}"
            mimic_text = f" · ↳ {joint.mimic_joint} {mode}"
        drive_text = ""
        if joint.drive_source_joint:
            rpm = joint.drive_max_velocity * 60.0 / (2.0 * math.pi)
            direction = " 반전" if joint.drive_reverse else ""
            drive_text = (
                f" · 주행 ↳ {joint.drive_source_joint} ±{rpm:g} RPM{direction}"
            )
        if joint.type == "prismatic":
            value0 = float(joint.lower or 0.0) * 1000.0
            value1 = float(joint.upper or 0.0) * 1000.0
            return f"직선 {axis_text} · 0:{value0:g} → 1:{value1:g} mm{mimic_text}"
        if joint.type == "revolute":
            value0 = math.degrees(float(joint.lower or 0.0))
            value1 = math.degrees(float(joint.upper or 0.0))
            return f"회전 {axis_text} · 0:{value0:g} → 1:{value1:g}°{mimic_text}"
        if joint.type == "continuous":
            return f"연속 회전 {axis_text}{mimic_text}{drive_text}"
        return joint.type

    def _rebuild_viewport(
        self,
        selection: Iterable[str] = (),
        *,
        include_assigned: bool = False,
    ) -> None:
        if self.project is None:
            self.viewport.clear()
            return
        visible_parts = [
            part
            for part in self.project.parts.values()
            if part.visible
            and not (
                self._hide_assigned_parts
                and not include_assigned
                and part.link_name is not None
            )
        ]
        self.viewport.set_parts(visible_parts)
        self._apply_viewport_color_overrides(visible_parts)
        self._update_scene_transforms()
        visible_ids = {part.id for part in visible_parts}
        self.viewport.set_selected([part_id for part_id in selection if part_id in visible_ids])

    def _apply_viewport_color_overrides(
        self,
        visible_parts: Iterable[Any] | None = None,
    ) -> None:
        """Apply normal display colors plus a red collision alarm override."""

        if self.project is None:
            return
        parts = list(visible_parts) if visible_parts is not None else [
            part
            for part in self.project.parts.values()
            if part.visible
        ]
        color_overrides: dict[str, tuple[float, float, float, float]] = {}
        if self.part_colors_action.isChecked():
            for part in parts:
                color_overrides[part.id] = _stable_part_display_color(
                    part.id,
                    float(part.color[3]),
                )
        for part in parts:
            if part.id in self._collision_highlight_part_ids:
                color_overrides[part.id] = (
                    *COLLISION_DISPLAY_RGB,
                    float(part.color[3]),
                )
        self.viewport.set_color_overrides(color_overrides)

    def _update_scene_transforms(self) -> None:
        if self.project is None or not self.project.links:
            return
        try:
            zero_fk = self.project.forward_kinematics(zero=True)
            current_fk = self.project.forward_kinematics()
        except ProjectValidationError:
            return
        deltas: dict[str, np.ndarray] = {}
        for part in self.project.parts.values():
            if not part.visible or part.link_name not in current_fk:
                continue
            deltas[part.id] = current_fk[part.link_name] @ np.linalg.inv(zero_fk[part.link_name])
        self.viewport.update_part_transforms(deltas)
        self._refresh_axis_marker(current_fk)

    def _refresh_axis_marker(self, current_fk: dict[str, np.ndarray] | None = None) -> None:
        if self.project is None or self.current_joint is None:
            self.viewport.clear_axis_marker()
            return
        try:
            joint = self.project.joint(self.current_joint)
            current_fk = current_fk or self.project.forward_kinematics()
            frame = current_fk[joint.parent] @ joint.origin_transform()
            direction = frame[:3, :3] @ (joint.axis / np.linalg.norm(joint.axis))
            rotational = joint.type in {"revolute", "continuous"}
            if rotational:
                # A rotation axis is the line through the joint origin, not a
                # convenient arrow placed at the selected link's BBox center.
                marker_origin = frame[:3, 3]
            else:
                marker_origin = self._current_link_bbox_center(current_fk)
                if marker_origin is None:
                    marker_origin = frame[:3, 3]
            self.viewport.set_axis_marker(
                marker_origin,
                direction,
                self._scene_scale * (0.20 if rotational else 0.10),
                bidirectional=rotational,
            )
        except Exception:
            self.viewport.clear_axis_marker()

    def _current_link_bbox_center(
        self,
        current_fk: dict[str, np.ndarray],
    ) -> np.ndarray | None:
        if (
            self.project is None
            or self.current_link is None
            or self.current_link not in self.project.links
            or self.current_link not in current_fk
        ):
            return None
        link = self.project.links[self.current_link]
        zero_fk = self.project.forward_kinematics(zero=True)
        if self.current_link not in zero_fk:
            return None
        delta = current_fk[self.current_link] @ np.linalg.inv(zero_fk[self.current_link])
        vertices = [
            apply_transform(self.project.parts[part_id].vertices_zero, delta)
            for part_id in link.part_ids
            if part_id in self.project.parts
            and len(self.project.parts[part_id].vertices_zero)
        ]
        if not vertices:
            return None
        minimum = np.min([value.min(axis=0) for value in vertices], axis=0)
        maximum = np.max([value.max(axis=0) for value in vertices], axis=0)
        return (minimum + maximum) * 0.5

    def _set_demo_playing(self, playing: bool) -> None:
        if not playing:
            if self._animation_mode == "auto":
                self._stop_demo_animation(restore=True)
            return
        if self._animation_mode is not None:
            self._stop_demo_animation(restore=True)
        if self.project is None:
            self.viewport.set_animation_playing(False)
            return
        movable: list[JointSpec] = []
        drive_targets = [
            joint for joint in self.project.joints if joint.drive_source_joint
        ]
        drive_target_names = {joint.name for joint in drive_targets}
        for joint in self.project.joints:
            if joint.mimic_joint:
                continue
            if joint.name in drive_target_names:
                continue
            if joint.type in {"prismatic", "revolute"}:
                lower = joint.lower
                upper = joint.upper
            elif joint.type == "continuous":
                lower, upper = -math.pi, math.pi
            else:
                continue
            if (
                lower is not None
                and upper is not None
                and math.isfinite(float(lower))
                and math.isfinite(float(upper))
                and not math.isclose(float(lower), float(upper))
            ):
                movable.append(joint)
        if not movable and not drive_targets:
            self.viewport.set_animation_playing(False)
            self.statusBar().showMessage("재생할 가동 관절이 없습니다.", 5000)
            return

        self._animation_mode = "auto"
        self._demo_joints = movable
        self._demo_drive_joints = drive_targets
        self._demo_original_positions = {
            joint.name: float(joint.position) for joint in self.project.joints
        }
        self._demo_started_at = time.perf_counter()
        self._demo_last_tick = self._demo_started_at
        self.viewport.set_control_animation_playing(False)
        self.viewport.set_animation_playing(True)
        self._demo_timer.start()
        if self._collision_precheck_count:
            self.statusBar().showMessage(
                f"⚠ 실제 메시 침투 충돌 {self._collision_precheck_count}건 · "
                "'검증 및 알림'을 확인하세요."
            )
        elif drive_targets:
            self.statusBar().showMessage(
                "자동 재생 중 · 모든 독립 입력과 연결된 바퀴를 자동으로 시험합니다."
            )
        self._advance_demo_animation()

    def _operator_control_descriptors(self) -> list[dict[str, Any]]:
        if self.project is None:
            return []
        mimic_sources = {
            str(joint.mimic_joint)
            for joint in self.project.joints
            if joint.mimic_joint
        }
        drive_sources = {
            str(joint.drive_source_joint)
            for joint in self.project.joints
            if joint.drive_source_joint
        }
        descriptors: list[dict[str, Any]] = []
        for joint in self.project.joints:
            # Operator mode must expose every independently controllable joint.
            # Mimic followers and velocity-driven targets are omitted because
            # changing their source already updates them.
            if joint.mimic_joint or joint.drive_source_joint:
                continue
            limits = self.project._joint_preview_limits(joint)
            if limits is None or math.isclose(limits[0], limits[1]):
                continue
            if joint.type == "prismatic":
                display_scale, units = 1000.0, " mm"
            else:
                display_scale, units = 180.0 / math.pi, "°"
            roles: list[str] = []
            if joint.name in mimic_sources:
                roles.append("연동 입력")
            if joint.name in drive_sources:
                roles.append("주행 레버")
            if not roles:
                roles.append(
                    {
                        "prismatic": "직선 관절",
                        "revolute": "회전 관절",
                        "continuous": "연속 회전",
                    }.get(joint.type, "관절 조작")
                )
            descriptors.append(
                {
                    "name": joint.name,
                    "label": f"{joint.child} ({joint.name})",
                    "role": " / ".join(roles),
                    "lower": limits[0],
                    "upper": limits[1],
                    "value": joint.position,
                    "display_scale": display_scale,
                    "units": units,
                    "hint": (
                        "후진 ← 중립 → 전진"
                        if joint.name in drive_sources
                        else "상태 0 ← 가운데 → 상태 1"
                    ),
                }
            )
        return descriptors

    def _set_control_playing(self, playing: bool) -> None:
        if not playing:
            if self._animation_mode == "control":
                self._stop_demo_animation(restore=True)
            return
        if self._animation_mode is not None:
            self._stop_demo_animation(restore=True)
        if self.project is None:
            self.viewport.set_control_animation_playing(False)
            return
        controls = self._operator_control_descriptors()
        if not controls:
            self.viewport.set_control_animation_playing(False)
            self.statusBar().showMessage(
                "조작 가능한 독립 가동 관절이 없습니다.",
                7000,
            )
            return

        self._animation_mode = "control"
        self._demo_joints = []
        self._demo_drive_joints = [
            joint for joint in self.project.joints if joint.drive_source_joint
        ]
        self._demo_original_positions = {
            joint.name: float(joint.position) for joint in self.project.joints
        }
        self._demo_started_at = time.perf_counter()
        self._demo_last_tick = self._demo_started_at
        self.viewport.set_operator_controls(controls)
        self.viewport.set_animation_playing(False)
        self.viewport.set_control_animation_playing(True)
        self._set_operator_layout_active(True)
        if self._demo_drive_joints:
            self._demo_timer.start()
            self._advance_demo_animation()
        if self._collision_precheck_count:
            self.statusBar().showMessage(
                f"⚠ 실제 메시 침투 충돌 {self._collision_precheck_count}건 · "
                "조작 정지 후 '검증 및 알림'을 확인하세요."
            )
        else:
            self.statusBar().showMessage(
                f"조작 Play · 3D 화면에서 {len(controls)}개 독립 관절을 직접 시험합니다."
            )

    def _set_operator_control_value(self, joint_name: str, value: float) -> None:
        if self.project is None or self._animation_mode != "control":
            return
        try:
            self.project.set_joint_position(joint_name, value)
        except (KeyError, ValueError):
            return
        self._update_scene_transforms()

    def _set_operator_layout_active(self, active: bool) -> None:
        if active:
            if self._operator_panel_visibility is None:
                self._operator_panel_visibility = (
                    not self.left_panel.isHidden(),
                    not self.inspector_dock.isHidden(),
                    not self.issues_dock.isHidden(),
                )
            self.left_panel.hide()
            self.inspector_dock.hide()
            self.issues_dock.hide()
            return
        if self._operator_panel_visibility is None:
            return
        left_visible, inspector_visible, issues_visible = (
            self._operator_panel_visibility
        )
        self.left_panel.setVisible(left_visible)
        self.inspector_dock.setVisible(inspector_visible)
        self.issues_dock.setVisible(issues_visible)
        self._operator_panel_visibility = None

    def _advance_demo_animation(self) -> None:
        if self.project is None or (
            not (self._demo_joints or self._demo_drive_joints)
            and self._animation_mode != "control"
        ):
            self._stop_demo_animation(restore=False)
            return
        now = time.perf_counter()
        elapsed = now - self._demo_started_at
        delta_seconds = min(max(now - self._demo_last_tick, 0.0), 0.1)
        self._demo_last_tick = now

        for index, joint in enumerate(self._demo_joints):
            if joint.type == "continuous":
                lower, upper = -math.pi, math.pi
            else:
                lower = float(joint.lower or 0.0)
                upper = float(joint.upper or 0.0)
            phase = (elapsed / 6.0 + index * 0.17) % 2.0
            ratio = phase if phase <= 1.0 else 2.0 - phase
            joint.position = joint.clamp(lower + (upper - lower) * ratio)

        resolved = self.project.apply_mimic_positions()
        for joint in self._demo_drive_joints:
            angular_velocity = self.project.drive_velocity(joint, resolved)
            joint.position += angular_velocity * delta_seconds

        self._update_scene_transforms()

    def _stop_demo_animation(self, *, restore: bool) -> None:
        self._demo_timer.stop()
        if restore and self.project is not None:
            for joint in self.project.joints:
                if joint.name in self._demo_original_positions:
                    joint.position = joint.clamp(
                        self._demo_original_positions[joint.name]
                    )
            self.project.apply_mimic_positions()
            if self._demo_original_positions:
                self._update_scene_transforms()
        self._demo_joints = []
        self._demo_drive_joints = []
        self._demo_original_positions = {}
        self._animation_mode = None
        if hasattr(self, "viewport"):
            self.viewport.set_animation_playing(False)
            self.viewport.set_control_animation_playing(False)
            self.viewport.set_operator_controls([])
        self._set_operator_layout_active(False)
        if hasattr(self, "joint_editor"):
            self._refresh_editor()

    def _refresh_editor(self) -> None:
        if self.project is None:
            self.joint_editor.set_joint(None)
            return
        self.joint_editor.set_link_names(self.project.links.keys())
        self.joint_editor.set_joint_specs(self.project.joints)
        try:
            joint = self.project.joint(self.current_joint) if self.current_joint else None
        except KeyError:
            joint = None
        self.joint_editor.set_joint(joint)

    def _refresh_issues(self) -> None:
        self.issue_list.clear()
        if self.project is None:
            return
        errors = self.project.validate(check_names=False)
        if errors:
            self._add_issues(errors, prefix="오류: ")
        warnings: list[str] = []
        for key in ("warnings", "step_warnings"):
            value = self.project.metadata.get(key, [])
            if isinstance(value, list):
                warnings.extend(str(item) for item in value)
        unassigned_count = sum(
            part.link_name is None for part in self.project.parts.values()
        )
        if unassigned_count:
            warnings.append(
                f"미할당 형상 {unassigned_count}개 · URDF 내보내기 전에 링크에 배정해야 합니다."
            )
        if not errors:
            for issue in self._collision_precheck_issues_cache:
                if issue.is_collision:
                    self._add_collision_issue(issue)
                else:
                    warnings.append(issue.message)
            if self._collision_check_pending:
                warnings.append("실제 삼각형 메시 충돌 검사 중…")
        self._add_issues(warnings, prefix="알림: ")
        if (
            not errors
            and not warnings
            and not self._collision_precheck_issues_cache
        ):
            self.issue_list.addItem("검증 통과")

    def _add_issues(self, messages: Iterable[str], prefix: str = "") -> None:
        for message in messages:
            self.issue_list.addItem(prefix + str(message))

    def _add_collision_issue(self, issue: _CollisionIssue) -> None:
        item = QListWidgetItem(issue.message)
        item.setForeground(QColor(224, 45, 45))
        font = item.font()
        font.setBold(True)
        item.setFont(font)
        if issue.part_ids:
            item.setData(ROLE_COLLISION_PARTS, issue.part_ids)
            item.setToolTip(
                "선택하면 이 충돌에 관련된 두 파트를 3D 화면에서 "
                "빨간색으로 강조합니다."
            )
        self.issue_list.addItem(item)

    def _issue_selection_changed(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        raw_part_ids = (
            current.data(ROLE_COLLISION_PARTS) if current is not None else None
        )
        if isinstance(raw_part_ids, (list, tuple, set)):
            part_ids = {
                str(part_id)
                for part_id in raw_part_ids
                if self.project is not None and str(part_id) in self.project.parts
            }
        else:
            part_ids = set()
        if part_ids == self._collision_highlight_part_ids:
            return
        self._collision_highlight_part_ids = part_ids
        self._apply_viewport_color_overrides()
        if part_ids:
            self.statusBar().showMessage(
                "충돌 항목 선택 · 관련 파트를 빨간색으로 강조했습니다.",
                7000,
            )

    def _format_collision_precheck_messages(
        self,
        current: Iterable[Any],
        motion: Iterable[Any],
        omitted: int,
    ) -> list[_CollisionIssue]:
        """Format exact mesh penetration results for the validation panel."""

        if self.project is None:
            return []
        adjacent = {
            frozenset((joint.parent, joint.child)) for joint in self.project.joints
        }
        current_by_parts: dict[frozenset[str], Any] = {}
        for candidate in current:
            if frozenset((candidate.link_a, candidate.link_b)) in adjacent:
                continue
            key = frozenset((candidate.part_a, candidate.part_b))
            current_by_parts.setdefault(key, candidate)

        motion_by_parts: dict[frozenset[str], Any] = {}
        for finding in motion:
            candidate = finding.candidate
            key = frozenset((candidate.part_a, candidate.part_b))
            motion_by_parts.setdefault(key, finding)

        total = len(current_by_parts) + len(
            set(motion_by_parts) - set(current_by_parts)
        )
        self._collision_precheck_count = total
        issues: list[_CollisionIssue] = []
        if total:
            issues.append(
                _CollisionIssue(
                    f"⚠ 메시 충돌 {total}건 · 아래 항목을 선택하면 "
                    "관련 파트를 빨간색으로 표시합니다."
                )
            )
        displayed = 0
        for candidate in list(current_by_parts.values())[:50]:
            part_a = self.project.parts[candidate.part_a].name
            part_b = self.project.parts[candidate.part_b].name
            issues.append(
                _CollisionIssue(
                    f"⚠ 충돌 [현재 자세] {candidate.link_a} ↔ "
                    f"{candidate.link_b} · {part_a} ({candidate.part_a}) ↔ "
                    f"{part_b} ({candidate.part_b})",
                    (candidate.part_a, candidate.part_b),
                )
            )
            displayed += 1
        for part_key, finding in motion_by_parts.items():
            if displayed >= 50:
                break
            if part_key in current_by_parts:
                continue
            if finding.joint_name is None:
                joint_name = "전체 가동 관절"
                sample = f"동시 {finding.position * 100.0:.0f}%"
            else:
                joint = self.project.joint(finding.joint_name)
                joint_name = joint.name
                if joint.type == "prismatic":
                    sample = f"{finding.position * 1000.0:.1f} mm"
                else:
                    sample = f"{math.degrees(finding.position):.1f}°"
            candidate = finding.candidate
            part_a = self.project.parts[candidate.part_a].name
            part_b = self.project.parts[candidate.part_b].name
            issues.append(
                _CollisionIssue(
                    f"⚠ 충돌 [가동 범위] {candidate.link_a} ↔ "
                    f"{candidate.link_b} · {joint_name}={sample} · "
                    f"{part_a} ({candidate.part_a}) ↔ "
                    f"{part_b} ({candidate.part_b})",
                    (candidate.part_a, candidate.part_b),
                )
            )
            displayed += 1
        hidden = total - displayed
        if hidden > 0:
            issues.append(_CollisionIssue(f"⚠ 충돌 {hidden}건이 더 있습니다."))
        if omitted:
            issues.append(
                _CollisionIssue(
                    f"관절이 많아 뒤의 {omitted}개는 가동 범위 샘플 "
                    "검사에서 제외됐습니다.",
                    is_collision=False,
                )
            )
        if total:
            issues.append(
                _CollisionIssue(
                    "충돌 결과는 OBB 후보를 실제 삼각형 메시 침투로 "
                    "다시 검증한 결과입니다.",
                    is_collision=False,
                )
            )
        return issues

    def _schedule_collision_precheck(self) -> None:
        """Run dense mesh collision checks without freezing a large scene."""

        if self.project is None:
            return
        if self.project.validate(check_names=False):
            self._collision_precheck_count = 0
            self._collision_precheck_issues_cache = []
            self._collision_highlight_part_ids = set()
            self._apply_viewport_color_overrides()
            self._collision_check_pending = False
            self._refresh_issues()
            return
        if self._collision_checker is None:
            self._collision_checker = MeshCollisionChecker(self.project)
        self._collision_highlight_part_ids = set()
        self._apply_viewport_color_overrides()
        if self._collision_worker_active:
            self._collision_rescan_requested = True
            self._collision_check_pending = True
            self._refresh_issues()
            return

        project = self.project
        checker = self._collision_checker
        tolerance = max(self._scene_scale * 1.0e-6, 1.0e-7)

        def operation() -> tuple[list[Any], list[Any], int]:
            return checker.sampled_self_collisions(
                samples_per_joint=3,
                max_joints=32,
                contact_tolerance=tolerance,
            )

        def apply_result(result: tuple[list[Any], list[Any], int]) -> None:
            current, motion, omitted = result
            self._collision_highlight_part_ids = set()
            self._collision_precheck_issues_cache = (
                self._format_collision_precheck_messages(
                    current,
                    motion,
                    omitted,
                )
            )
            self._collision_check_pending = False
            self._apply_viewport_color_overrides()
            self._refresh_issues()
            if self._collision_precheck_count:
                self.statusBar().showMessage(
                    f"실제 메시 충돌 검사 완료 · "
                    f"침투 충돌 {self._collision_precheck_count}건",
                    10000,
                )
            else:
                self.statusBar().showMessage(
                    "실제 메시 충돌 검사 완료 · 침투 충돌 없음",
                    10000,
                )

        triangle_count = sum(
            len(part.triangles) for part in project.parts.values()
        )
        if triangle_count <= 50000:
            try:
                apply_result(operation())
            except (ProjectValidationError, ValueError, np.linalg.LinAlgError):
                self._collision_precheck_count = 0
                self._collision_highlight_part_ids = set()
                self._apply_viewport_color_overrides()
                self._collision_precheck_issues_cache = [
                    _CollisionIssue(
                        "실제 메시 충돌 검사를 실행하지 못했습니다.",
                        is_collision=False,
                    )
                ]
                self._refresh_issues()
            return

        self._collision_worker_active = True
        self._collision_check_pending = True
        self._refresh_issues()
        worker = _FunctionWorker(operation)
        self._workers.add(worker)

        def cleanup() -> None:
            self._workers.discard(worker)
            self._collision_worker_active = False

        def success(result: tuple[list[Any], list[Any], int]) -> None:
            cleanup()
            if self.project is project:
                apply_result(result)
            if self._collision_rescan_requested:
                self._collision_rescan_requested = False
                self._schedule_collision_precheck()

        def failure(_details: str) -> None:
            cleanup()
            if self.project is project:
                self._collision_check_pending = False
                self._collision_precheck_count = 0
                self._collision_highlight_part_ids = set()
                self._apply_viewport_color_overrides()
                self._collision_precheck_issues_cache = [
                    _CollisionIssue(
                        "실제 메시 충돌 검사를 실행하지 못했습니다.",
                        is_collision=False,
                    )
                ]
                self._refresh_issues()
            if self._collision_rescan_requested:
                self._collision_rescan_requested = False
                self._schedule_collision_precheck()

        worker.signals.finished.connect(success)
        worker.signals.failed.connect(failure)
        QThreadPool.globalInstance().start(worker)

    # ---------- selection ----------
    def _walk_part_tree(self) -> Iterable[QTreeWidgetItem]:
        def visit(item: QTreeWidgetItem) -> Iterable[QTreeWidgetItem]:
            yield item
            for child_index in range(item.childCount()):
                yield from visit(item.child(child_index))

        for index in range(self.part_tree.topLevelItemCount()):
            yield from visit(self.part_tree.topLevelItem(index))

    def _part_ids_below(self, item: QTreeWidgetItem) -> list[str]:
        part_id = item.data(0, ROLE_ID)
        if part_id is not None:
            return [str(part_id)]
        result: list[str] = []
        for index in range(item.childCount()):
            result.extend(self._part_ids_below(item.child(index)))
        return result

    def _selected_part_ids(self) -> list[str]:
        # The tree is the canonical selection model. Hidden parts have no VTK
        # actor, but users still need to select/reassign them from the list.
        if hasattr(self, "part_tree"):
            selected: list[str] = []
            for item in self.part_tree.selectedItems():
                selected.extend(self._part_ids_below(item))
            return list(dict.fromkeys(selected))
        if hasattr(self, "viewport"):
            return self.viewport.selected_ids()
        return []

    def _set_selected_parts(self, part_ids: Iterable[str]) -> None:
        identifiers = list(dict.fromkeys(str(item) for item in part_ids))
        self._selection_guard = True
        try:
            self.viewport.set_selected(identifiers)
            selected = set(identifiers)
            for item in self._walk_part_tree():
                item.setSelected(item.data(0, ROLE_ID) in selected)
        finally:
            self._selection_guard = False
        self._update_selection_summary(identifiers)

    def _viewport_selection_changed(self, part_ids: list[str]) -> None:
        if self._selection_guard:
            return
        self._selection_guard = True
        try:
            selected = set(part_ids)
            for item in self._walk_part_tree():
                item.setSelected(item.data(0, ROLE_ID) in selected)
        finally:
            self._selection_guard = False
        self._update_selection_summary(part_ids)

    def _part_tree_selection_changed(self) -> None:
        if self._selection_guard:
            return
        identifiers = self._selected_part_ids()
        self._selection_guard = True
        try:
            self.viewport.set_selected(identifiers)
        finally:
            self._selection_guard = False
        self._update_selection_summary(identifiers)

    def _part_tree_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._tree_guard or self.project is None or column != 0:
            return
        part_id = item.data(0, ROLE_ID)
        if part_id in self.project.parts:
            self.project.parts[part_id].visible = item.checkState(0) == Qt.CheckState.Checked
            selected = self._selected_part_ids()
            self._rebuild_viewport(selected)
            self._mark_dirty()

    def _update_selection_summary(self, part_ids: Iterable[str]) -> None:
        identifiers = list(part_ids)
        if self.project is None or not identifiers:
            self.selection_label.setText("선택된 형상 없음")
            return
        names = [self.project.parts[item].name for item in identifiers if item in self.project.parts]
        links = sorted(
            {self.project.parts[item].link_name or "미할당" for item in identifiers if item in self.project.parts}
        )
        preview = ", ".join(names[:3]) + (f" 외 {len(names) - 3}개" if len(names) > 3 else "")
        self.selection_label.setText(
            f"선택 {len(names)}개 · 링크 {', '.join(links)}\n{preview}"
        )

    def _link_tree_selection_changed(self) -> None:
        items = self.link_tree.selectedItems()
        if not items or self.project is None:
            return
        item = items[0]
        link_name = item.data(0, ROLE_ID)
        self.current_link = link_name
        self.current_joint = item.data(0, ROLE_JOINT)
        link = self.project.links.get(link_name)
        self._rebuild_current_link_parts()
        if link:
            if self._hide_assigned_parts:
                self._set_selected_parts([])
                self.selection_label.setText(
                    f"현재 링크 {link_name} · 형상 {len(link.part_ids)}개 숨김\n"
                    "Ctrl+H로 사용한 파츠를 표시할 수 있습니다."
                )
            else:
                self._set_selected_parts(link.part_ids)
        self._refresh_editor()
        self._update_scene_transforms()

    def _rebuild_current_link_parts(self) -> None:
        if not hasattr(self, "link_parts_list"):
            return
        self._link_parts_guard = True
        try:
            self.link_parts_list.clear()
            if self.project is None or self.current_link not in self.project.links:
                self.link_parts_label.setText("현재 링크 형상 (링크 선택 필요)")
                return
            link = self.project.links[self.current_link]
            self.link_parts_label.setText(
                f"현재 링크 형상 · {self.current_link} ({len(link.part_ids)}개)"
            )
            for part_id in link.part_ids:
                part = self.project.parts.get(part_id)
                if part is None:
                    continue
                self.link_parts_list.addItem(part.name)
                item = self.link_parts_list.item(self.link_parts_list.count() - 1)
                item.setData(ROLE_ID, part.id)
                item.setToolTip(f"파트 ID: {part.id}")
        finally:
            self._link_parts_guard = False

    def _link_parts_selection_changed(self) -> None:
        if self._link_parts_guard:
            return
        identifiers = [
            str(item.data(ROLE_ID))
            for item in self.link_parts_list.selectedItems()
            if item.data(ROLE_ID) is not None
        ]
        self._set_selected_parts(identifiers)

    def _select_link_item(self, link_name: str) -> None:
        def visit(item: QTreeWidgetItem) -> QTreeWidgetItem | None:
            if item.data(0, ROLE_ID) == link_name:
                return item
            for child_index in range(item.childCount()):
                found = visit(item.child(child_index))
                if found:
                    return found
            return None

        for index in range(self.link_tree.topLevelItemCount()):
            found = visit(self.link_tree.topLevelItem(index))
            if found:
                self.link_tree.setCurrentItem(found)
                return

    # ---------- topology editing ----------
    def new_manual_tree(self) -> None:
        if self.project is None:
            return

        root_name, accepted = QInputDialog.getText(
            self,
            "새 수동 링크 트리",
            (
                "Base 링크 이름\n"
                "빈 Base를 만들고 모든 형상을 미할당 상태로 되돌립니다."
            ),
            text="base_link",
        )
        if not accepted or not root_name.strip():
            return
        if self.project.links or self.project.joints:
            answer = QMessageBox.question(
                self,
                "링크 트리 새로 만들기",
                "기존 링크와 관절 구성을 지우고 새 Base에서 시작할까요?\n"
                "STEP 형상 자체는 지워지지 않습니다.",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        safe_root = sanitize_name(root_name.strip(), "base_link")
        backup = copy.deepcopy(self.project)
        try:
            self.project.links.clear()
            self.project.joints.clear()
            self.project.root_link = None
            for part in self.project.parts.values():
                part.link_name = None
            self.project.create_link(safe_root)
            self.project.root_link = safe_root
            self.project.metadata.pop("mechanisms", None)
            errors = self.project.validate(check_names=False)
            if errors:
                raise ProjectValidationError(errors)
            self.project.metadata["manual_tree"] = True
        except Exception as exc:
            self.project = backup
            self._rebuild_all(preserve_selection=False)
            QMessageBox.critical(self, "새 트리 생성 실패", str(exc))
            return

        self.current_link = safe_root
        self.current_joint = None
        self._hide_assigned_parts = True
        self.hide_assigned_action.blockSignals(True)
        self.hide_assigned_action.setChecked(True)
        self.hide_assigned_action.blockSignals(False)
        self._update_assigned_visibility_controls()
        self._mark_dirty()
        self._rebuild_all(preserve_selection=False)
        self._select_link_item(safe_root)
        self.left_tabs.setCurrentIndex(1)
        unassigned = len(self.project.parts)
        self.statusBar().showMessage(
            f"새 트리 생성 · Base {safe_root} · 미할당 형상 {unassigned}개",
            10000,
        )

    def create_link_from_selection(self) -> None:
        self.add_child_link_from_selection()

    def open_mechanism_wizard(self) -> None:
        if self.project is None:
            return
        if self._new_link_dialog is not None:
            self._new_link_dialog.show()
            self._new_link_dialog.raise_()
            self._new_link_dialog.activateWindow()
            self.statusBar().showMessage(
                "열려 있는 자식 링크 도구 창에서 설정을 마치거나 취소하세요.",
                5000,
            )
            return
        part_ids = self._selected_part_ids()
        if not part_ids:
            self.left_tabs.setCurrentIndex(0)
            self.statusBar().showMessage(
                "먼저 3D 또는 형상 목록에서 마법사로 구성할 움직이는 형상을 선택하세요.",
                10000,
            )
            return
        parent = (
            self.current_link
            if self.current_link in self.project.links
            else self.project.root_link
        )
        if parent not in self.project.links:
            QMessageBox.information(
                self,
                "Base 링크 필요",
                "먼저 '전체 새로 시작'으로 Base 링크를 만드세요.",
            )
            return
        wizard = MechanismWizard(
            self.project.links.keys(),
            self.project.joints,
            default_parent=parent,
            selected_count=len(part_ids),
            parent=self,
        )
        if wizard.exec() != QDialog.DialogCode.Accepted:
            return
        self._pending_mechanism_preset = wizard.values()
        self.add_child_link_from_selection()

    def add_child_link_from_selection(self) -> None:
        if self.project is None:
            return
        if self._new_link_dialog is not None:
            self._new_link_dialog.show()
            self._new_link_dialog.raise_()
            self._new_link_dialog.activateWindow()
            self.statusBar().showMessage(
                "열려 있는 자식 링크 도구 창에서 설정을 마치거나 취소하세요.",
                5000,
            )
            return
        mechanism_preset = self._pending_mechanism_preset
        self._pending_mechanism_preset = None
        part_ids = self._selected_part_ids()
        if not part_ids:
            self.left_tabs.setCurrentIndex(0)
            self.statusBar().showMessage(
                "먼저 3D 또는 형상 목록에서 핸들처럼 움직일 자식 형상을 선택하세요.",
                10000,
            )
            return
        parent = (
            self.current_link
            if self.current_link in self.project.links
            else self.project.root_link
        )
        if (
            mechanism_preset is not None
            and mechanism_preset.get("parent") in self.project.links
        ):
            parent = str(mechanism_preset["parent"])
        if parent not in self.project.links:
            QMessageBox.information(
                self,
                "Base 링크 필요",
                "먼저 '전체 새로 시작'으로 Base 링크를 만드세요.",
            )
            return
        dialog = NewLinkDialog(
            self.project.links.keys(),
            default_parent=parent,
            selected_count=len(part_ids),
            lock_parent=False,
            parent=self,
        )
        if mechanism_preset is not None:
            dialog.apply_mechanism_preset(mechanism_preset)
        previous_selection = list(part_ids)
        project_at_start = self.project
        self._new_link_dialog = dialog

        def preview_parent(parent_name: str) -> None:
            self._preview_new_link_candidates(dialog, parent_name, part_ids)

        def preview_axis(axis: Iterable[float]) -> None:
            name = dialog.matching_candidate(axis)
            if name is not None:
                self.viewport.highlight_candidate_axis(name)

        def preview_motion() -> None:
            self._preview_new_link_motion(
                dialog,
                dialog.parent_combo.currentText(),
                part_ids,
            )

        def toggle_axis_edit(enabled: bool) -> None:
            if enabled:
                preview_motion()
                self.statusBar().showMessage(
                    "노란 축의 양 끝점을 끌어 방향을, 선 가운데를 끌어 관절 중심을 조정하세요."
                )
            else:
                self.viewport.clear_axis_edit_handles()

        def axis_handle_changed(
            origin_world: Iterable[float],
            direction_world: Iterable[float],
        ) -> None:
            if self.project is not project_at_start:
                return
            parent_name = dialog.parent_combo.currentText()
            if parent_name not in self.project.links:
                return
            try:
                parent_current = self.project.forward_kinematics()[parent_name]
                inverse_parent = np.linalg.inv(parent_current)
                origin_parent = apply_transform(
                    np.asarray(tuple(origin_world), dtype=float).reshape(1, 3),
                    inverse_parent,
                )[0]
                axis_parent = (
                    inverse_parent[:3, :3]
                    @ np.asarray(tuple(direction_world), dtype=float)
                )
            except (KeyError, ValueError, np.linalg.LinAlgError):
                return
            dialog.set_axis_from_3d(origin_parent, axis_parent)

        dialog.parentPreviewRequested.connect(preview_parent)
        dialog.axisPreviewRequested.connect(preview_axis)
        dialog.motionPreviewRequested.connect(preview_motion)
        dialog.axisEditModeChanged.connect(toggle_axis_edit)
        self.viewport.candidateAxisPicked.connect(dialog.select_candidate_axis)
        self.viewport.axisHandlesChanged.connect(axis_handle_changed)
        dialog.finished.connect(
            lambda result: self._finish_new_link_dialog(
                dialog,
                result,
                part_ids,
                previous_selection,
                project_at_start,
                axis_handle_changed,
                mechanism_preset,
            )
        )
        self._rebuild_viewport(part_ids, include_assigned=True)
        preview_parent(parent)
        dialog.show()
        dialog.fit_to_available_screen()
        if isinstance(dialog, QDialog):
            top_right = self.mapToGlobal(self.rect().topRight())
            screen = self.screen() or QApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                target_x = top_right.x() - dialog.width() - 24
                target_y = top_right.y() + 72
                dialog.move(
                    max(
                        available.left() + 12,
                        min(target_x, available.right() - dialog.width() - 12),
                    ),
                    max(
                        available.top() + 12,
                        min(target_y, available.bottom() - dialog.height() - 12),
                    ),
                )
        dialog.raise_()
        dialog.activateWindow()
        self.statusBar().showMessage(
            "도구 창을 열어 둔 채 3D 뷰를 움직이세요. A/B/C 선을 직접 클릭해 축을 고를 수 있습니다."
        )

    def _finish_new_link_dialog(
        self,
        dialog: NewLinkDialog,
        result: int,
        part_ids: Iterable[str],
        previous_selection: Iterable[str],
        project_at_start: RobotProject,
        axis_handle_changed: Callable[[Iterable[float], Iterable[float]], None],
        mechanism_preset: dict[str, Any] | None = None,
    ) -> None:
        """Close a modeless joint-authoring session and optionally create it."""

        if dialog is not self._new_link_dialog:
            return
        self._new_link_dialog = None
        try:
            self.viewport.candidateAxisPicked.disconnect(dialog.select_candidate_axis)
        except (RuntimeError, TypeError):
            pass
        try:
            self.viewport.axisHandlesChanged.disconnect(axis_handle_changed)
        except (RuntimeError, TypeError):
            pass
        self.viewport.clear_axis_edit_handles(render=False)
        self.viewport.clear_candidate_axes(render=False)
        self._rebuild_viewport(previous_selection)
        dialog.deleteLater()
        if (
            result != QDialog.DialogCode.Accepted
            or self.project is not project_at_start
        ):
            return
        try:
            values = dialog.values()
        except ValueError as exc:
            QMessageBox.warning(self, "링크 설정 오류", str(exc))
            return
        if not values["link_name"]:
            QMessageBox.warning(self, "링크 설정 오류", "링크 이름을 입력하세요.")
            return
        link_name = sanitize_name(values["link_name"], "moving_link")
        joint_options: dict[str, Any] = {}
        if mechanism_preset is not None:
            joint_options.update(
                {
                    "effort": float(mechanism_preset.get("effort", 100.0)),
                    "velocity": float(mechanism_preset.get("velocity", 0.25)),
                    "damping": float(mechanism_preset.get("damping", 0.0)),
                    "friction": float(mechanism_preset.get("friction", 0.0)),
                }
            )
            source_mode = mechanism_preset.get("source_mode")
            source_joint = mechanism_preset.get("source_joint")
            if source_mode == "mimic" and source_joint:
                joint_options.update(
                    {
                        "mimic_joint": str(source_joint),
                        "mimic_auto": True,
                        "mimic_reverse": bool(mechanism_preset.get("reverse", False)),
                    }
                )
            elif source_mode == "drive" and source_joint:
                joint_options.update(
                    {
                        "drive_source_joint": str(source_joint),
                        "drive_max_velocity": (
                            float(mechanism_preset.get("max_rpm", 60.0))
                            * 2.0
                            * math.pi
                            / 60.0
                        ),
                        "drive_deadband": float(
                            mechanism_preset.get("deadband", 0.03)
                        ),
                        "drive_reverse": bool(mechanism_preset.get("reverse", False)),
                    }
                )
        created = self._create_moving_link(
            link_name,
            part_ids,
            parent=values["parent"],
            joint_name=f"{link_name}_joint",
            axis=values["axis"],
            origin_xyz=values.get("origin_xyz"),
            joint_type=values["joint_type"],
            lower=values["lower"],
            upper=values["upper"],
            allow_empty=False,
            joint_options=joint_options,
        )
        if created and mechanism_preset is not None:
            self._record_mechanism_metadata(
                mechanism_preset,
                link_name=link_name,
                joint_name=f"{link_name}_joint",
            )
            self._mark_dirty()
        if created:
            self.left_tabs.setCurrentIndex(1)

    def _record_mechanism_metadata(
        self,
        values: dict[str, Any],
        *,
        link_name: str,
        joint_name: str,
    ) -> None:
        if self.project is None:
            return
        mechanisms = self.project.metadata.get("mechanisms")
        if not isinstance(mechanisms, list):
            mechanisms = []
            self.project.metadata["mechanisms"] = mechanisms
        mechanisms[:] = [
            item
            for item in mechanisms
            if not isinstance(item, dict) or item.get("joint") != joint_name
        ]
        mechanisms.append(
            {
                "type": str(values.get("preset") or "custom"),
                "title": str(values.get("title") or "대표 기구"),
                "link": link_name,
                "joint": joint_name,
                "parent": str(values.get("parent") or ""),
                "state_0": str(values.get("state_0") or "상태 0"),
                "state_1": str(values.get("state_1") or "상태 1"),
                "simulation_role": str(values.get("simulation_role") or "position"),
                "source_joint": values.get("source_joint"),
                "reverse": bool(values.get("reverse", False)),
                "max_rpm": float(values.get("max_rpm", 0.0)),
                "deadband": float(values.get("deadband", 0.0)),
            }
        )

    def _preview_new_link_candidates(
        self,
        dialog: NewLinkDialog,
        parent: str,
        child_part_ids: Iterable[str],
    ) -> None:
        """Focus parent and child while deriving axes from the child geometry."""

        if self.project is None or parent not in self.project.links:
            return
        context_ids = [
            part_id
            for part_id in self.project.links[parent].part_ids
            if part_id in self.project.parts and self.project.parts[part_id].visible
        ]
        selected_ids = [
            part_id
            for part_id in child_part_ids
            if part_id in self.project.parts
            and self.project.parts[part_id].visible
        ]
        if not selected_ids:
            return
        try:
            zero_fk = self.project.forward_kinematics(zero=True)
            parent_zero = zero_fk[parent]
            center_zero = self._parts_center(selected_ids)
            center_parent = apply_transform(
                center_zero.reshape(1, 3),
                np.linalg.inv(parent_zero),
            )[0]
        except (KeyError, ValueError, np.linalg.LinAlgError):
            return

        child_parts = [self.project.parts[part_id] for part_id in selected_ids]
        parent_parts = [self.project.parts[part_id] for part_id in context_ids]
        cad_candidates = _cad_joint_axis_candidates(child_parts, parent_parts)
        child_vertices = [
            self.project.parts[part_id].vertices_zero
            for part_id in selected_ids
            if len(self.project.parts[part_id].vertices_zero)
        ]
        try:
            inverse_parent_rotation = np.linalg.inv(parent_zero[:3, :3])
        except np.linalg.LinAlgError:
            inverse_parent_rotation = parent_zero[:3, :3].T
        local_origins: dict[str, np.ndarray] = {}
        descriptions: dict[str, str] = {}
        if cad_candidates:
            local_candidates: dict[str, np.ndarray] = {}
            for name, candidate in zip("ABC", cad_candidates, strict=False):
                local_candidates[name] = (
                    inverse_parent_rotation @ candidate["direction"]
                )
                local_origins[name] = apply_transform(
                    candidate["origin"].reshape(1, 3),
                    np.linalg.inv(parent_zero),
                )[0]
                child_diameter = candidate["child_radius"] * 2000.0
                parent_radius = candidate["parent_radius"]
                if candidate["shared"] and parent_radius is not None:
                    parent_diameter = parent_radius * 2000.0
                    if abs(parent_diameter - child_diameter) < 0.2:
                        size_text = f"공통 Ø{child_diameter:.1f} mm"
                    else:
                        size_text = (
                            f"부모 Ø{parent_diameter:.1f} / 자식 Ø{child_diameter:.1f} mm"
                        )
                    descriptions[name] = f"{name}  STEP 결합 원통\n{size_text}"
                else:
                    descriptions[name] = (
                        f"{name}  자식 원통면\nØ{child_diameter:.1f} mm"
                    )
            center_parent = local_origins["A"].copy()
            recommended = "A"
        else:
            world_candidates = _geometry_principal_axes(child_vertices)
            local_candidates = {
                name: inverse_parent_rotation @ direction
                for name, direction in world_candidates.items()
            }
            if not local_candidates:
                local_candidates = {
                    name: np.eye(3, dtype=float)[:, index]
                    for index, name in enumerate("XYZ")
                }
            recommended = None
            descriptions = {
                "A": "A  BBox 가로 장축\n가장 긴 이동 방향",
                "B": "B  BBox 세로축\n두 번째 방향",
                "C": "C  BBox 두께축\n가장 얇은 방향",
            }

        dialog.set_candidate_origin(center_parent)
        dialog.set_axis_candidates(
            local_candidates,
            origins=local_origins,
            descriptions=descriptions,
            recommended=recommended,
        )
        self.viewport.clear_candidate_axes(render=False)
        self.viewport.set_isolated_parts([*context_ids, *selected_ids])
        self.viewport.set_selected(selected_ids)
        self._preview_new_link_motion(dialog, parent, selected_ids)
        self.viewport.frame_all()
        self.statusBar().showMessage(
            (
                "STEP의 부모·자식 원통면에서 실제 결합축 후보를 찾았습니다. "
                if cad_candidates
                else "STEP 원통 결합축이 없어 자식 형상의 통계 주축을 표시했습니다. "
            )
            + "회전 미리보기로 확인하고 필요하면 3D 축 편집을 사용하세요."
        )

    def _new_link_axis_length(
        self,
        parent: str,
        child_part_ids: Iterable[str],
    ) -> float:
        if self.project is None:
            return 0.01
        context_ids = [
            part_id
            for part_id in self.project.links[parent].part_ids
            if part_id in self.project.parts and self.project.parts[part_id].visible
        ]
        visible_ids = list(dict.fromkeys([*context_ids, *child_part_ids]))
        vertices = [
            self.project.parts[part_id].vertices_zero
            for part_id in visible_ids
            if part_id in self.project.parts
            if len(self.project.parts[part_id].vertices_zero)
        ]
        if vertices:
            lower = np.min([value.min(axis=0) for value in vertices], axis=0)
            upper = np.max([value.max(axis=0) for value in vertices], axis=0)
            return max(float(np.linalg.norm(upper - lower)) * 0.28, 0.01)
        return max(self._scene_scale * 0.15, 0.01)

    def _preview_new_link_motion(
        self,
        dialog: NewLinkDialog,
        parent: str,
        child_part_ids: Iterable[str],
    ) -> None:
        """Preview a proposed joint without modifying the project."""

        if self.project is None or parent not in self.project.links:
            return
        selected_ids = [
            part_id
            for part_id in child_part_ids
            if part_id in self.project.parts and self.project.parts[part_id].visible
        ]
        if not selected_ids:
            return
        try:
            zero_fk = self.project.forward_kinematics(zero=True)
            current_fk = self.project.forward_kinematics()
            parent_zero = zero_fk[parent]
            parent_current = current_fk[parent]
            origin_parent = dialog.origin_editor.value() / 1000.0
            origin_frame = np.eye(4, dtype=float)
            origin_frame[:3, 3] = origin_parent
            motion = np.eye(4, dtype=float)
            axis = np.asarray(dialog.axis_editor.value(), dtype=float)
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm > 1.0e-12:
                axis /= axis_norm
            position = dialog.preview_position_si()
            kind = dialog.type_combo.currentText()
            if kind in {"revolute", "continuous"} and axis_norm > 1.0e-12:
                motion = axis_angle_matrix(axis, position)
            elif kind == "prismatic" and axis_norm > 1.0e-12:
                motion[:3, 3] = axis * position
            delta = (
                parent_current
                @ origin_frame
                @ motion
                @ np.linalg.inv(parent_zero @ origin_frame)
            )
            center_current = apply_transform(
                origin_parent.reshape(1, 3),
                parent_current,
            )[0]
        except (KeyError, ValueError, np.linalg.LinAlgError):
            return

        self.viewport.update_part_transforms(
            {part_id: delta for part_id in selected_ids}
        )
        local_candidates = dialog.candidate_axes()
        local_candidate_origins = dialog.candidate_origins()
        directions = {
            name: parent_current[:3, :3] @ direction
            for name, direction in local_candidates.items()
        }
        candidate_origins = {
            name: apply_transform(origin.reshape(1, 3), parent_current)[0]
            for name, origin in local_candidate_origins.items()
        }
        if directions:
            selected_name = dialog.matching_candidate(axis) or ""
            axis_length = self._new_link_axis_length(parent, selected_ids)
            self.viewport.set_candidate_axes(
                center_current,
                directions,
                axis_length,
                selected=selected_name,
                selected_direction=(
                    parent_current[:3, :3] @ axis
                    if axis_norm > 1.0e-12 and kind != "fixed"
                    else None
                ),
                rotational=kind in {"revolute", "continuous"},
                candidate_origins=candidate_origins,
            )
            if dialog.axis_edit_button.isChecked() and axis_norm > 1.0e-12:
                self.viewport.set_axis_edit_handles(
                    center_current,
                    parent_current[:3, :3] @ axis,
                    axis_length,
                )
            else:
                self.viewport.clear_axis_edit_handles(render=False)
        else:
            self.viewport.clear_axis_edit_handles(render=False)

    def _create_moving_link(
        self,
        link_name: str,
        part_ids: Iterable[str],
        *,
        parent: str | None,
        joint_name: str,
        axis: Iterable[float],
        origin_xyz: Iterable[float] | None = None,
        joint_type: str = "prismatic",
        lower: float | None = None,
        upper: float | None = None,
        allow_empty: bool = False,
        joint_options: dict[str, Any] | None = None,
    ) -> bool:
        if self.project is None:
            return False
        identifiers = list(dict.fromkeys(part_ids))
        if not identifiers and not allow_empty:
            QMessageBox.information(self, "형상 선택 필요", "이 단계에 포함할 형상을 선택하세요.")
            return False
        if parent not in self.project.links:
            QMessageBox.warning(self, "부모 링크 없음", f"먼저 {parent!r} 링크를 구성하세요.")
            return False
        existing_errors = self.project.validate(check_names=False)
        if existing_errors:
            QMessageBox.warning(
                self,
                "기존 링크 트리 오류",
                "현재 링크 트리를 먼저 수정하세요.\n" + "\n".join(existing_errors),
            )
            return False
        if link_name in self.project.links:
            incoming = self.project.joint_for_child(link_name)
            if incoming is None or incoming.parent != parent or incoming.type != joint_type:
                QMessageBox.warning(
                    self,
                    "링크 이름 충돌",
                    f"{link_name!r} 링크가 이미 있지만 이 단계의 {parent!r} {joint_type} 관절로 연결되어 있지 않습니다. "
                    "기존 링크 이름을 바꾸거나 일반 링크 편집을 사용하세요.",
                )
                return False
            wrong_owner = [
                part_id
                for part_id in identifiers
                if self.project.parts[part_id].link_name not in {parent, link_name, None}
            ]
            if wrong_owner:
                owners = sorted(
                    {self.project.parts[item].link_name or "미할당" for item in wrong_owner}
                )
                QMessageBox.warning(
                    self,
                    "링크 계층 확인",
                    "선택 형상은 현재 단계 또는 부모 링크에 속해야 합니다. 현재 소속: "
                    + ", ".join(owners),
                )
                return False
            self.project.assign_parts(identifiers, link_name)
            self._after_topology_change(link_name)
            return True
        safe_joint_name = sanitize_name(joint_name, "joint")
        if any(joint.name == safe_joint_name for joint in self.project.joints):
            QMessageBox.warning(
                self,
                "관절 이름 충돌",
                f"{safe_joint_name!r} 관절이 이미 있습니다. 기존 관절 이름을 먼저 바꾸세요.",
            )
            return False
        wrong_owner = [
            part_id
            for part_id in identifiers
            if self.project.parts[part_id].link_name not in {parent, None}
        ]
        if wrong_owner:
            owners = sorted({self.project.parts[item].link_name or "미할당" for item in wrong_owner})
            QMessageBox.warning(
                self,
                "링크 계층 확인",
                "선택 형상은 부모 링크에 속해야 합니다. 현재 소속: " + ", ".join(owners),
            )
            return False
        try:
            parent_zero = self.project.forward_kinematics(zero=True)[parent]
        except Exception as exc:
            QMessageBox.warning(self, "링크 생성 실패", str(exc))
            return False
        if origin_xyz is not None:
            origin_parent = np.asarray(tuple(origin_xyz), dtype=float)
            if origin_parent.shape != (3,) or not np.all(np.isfinite(origin_parent)):
                QMessageBox.warning(
                    self,
                    "링크 생성 실패",
                    "후보 회전축 원점이 올바르지 않습니다.",
                )
                return False
        elif identifiers:
            origin_world = self._parts_center(identifiers)
            origin_parent = apply_transform(
                origin_world.reshape(1, 3), np.linalg.inv(parent_zero)
            )[0]
        else:
            origin_parent = np.zeros(3, dtype=float)
        previous_owners = {
            part_id: self.project.parts[part_id].link_name for part_id in identifiers
        }
        options = dict(joint_options or {})
        candidate_joint = JointSpec(
            name=safe_joint_name,
            type=joint_type,
            parent=parent,
            child=link_name,
            origin_xyz=origin_parent,
            axis=axis,
            lower=lower,
            upper=upper,
            effort=float(options.pop("effort", 100.0)),
            velocity=float(options.pop("velocity", 0.25)),
            damping=float(options.pop("damping", 0.0)),
            friction=float(options.pop("friction", 0.0)),
            **options,
        )
        self.project.create_link(link_name, identifiers)
        self.project.joints.append(candidate_joint)
        errors = self.project.validate(check_names=False)
        if errors:
            self.project.joints.remove(candidate_joint)
            self.project.links.pop(link_name, None)
            for part_id in identifiers:
                self.project.parts[part_id].link_name = None
            owners: dict[str | None, list[str]] = {}
            for part_id, owner in previous_owners.items():
                owners.setdefault(owner, []).append(part_id)
            for owner, owned_parts in owners.items():
                if owner is None:
                    continue
                self.project.assign_parts(owned_parts, owner)
            QMessageBox.warning(self, "링크 생성 실패", "\n".join(errors))
            return False
        self._after_topology_change(link_name)
        return True

    def _parts_center(self, part_ids: Iterable[str]) -> np.ndarray:
        if self.project is None:
            return np.zeros(3)
        vertices = [
            self.project.parts[item].vertices_zero
            for item in part_ids
            if item in self.project.parts and len(self.project.parts[item].vertices_zero)
        ]
        if not vertices:
            return np.zeros(3)
        minimum = np.min([value.min(axis=0) for value in vertices], axis=0)
        maximum = np.max([value.max(axis=0) for value in vertices], axis=0)
        return (minimum + maximum) / 2.0

    def assign_selection_to_link(self) -> None:
        if self.project is None:
            return
        part_ids = self._selected_part_ids()
        if not part_ids:
            return
        names = list(self.project.links)
        target, accepted = QInputDialog.getItem(
            self, "기존 링크로 이동", "대상 링크", names, editable=False
        )
        if accepted and target:
            self.project.assign_parts(part_ids, target)
            self._after_topology_change(target)

    def assign_selection_to_current_link(self) -> None:
        if self.project is None:
            return
        identifiers = self._selected_part_ids()
        if not identifiers:
            QMessageBox.information(self, "형상 선택 필요", "먼저 넣을 형상을 선택하세요.")
            return
        target = (
            self.current_link
            if self.current_link in self.project.links
            else self.project.root_link
        )
        if target not in self.project.links:
            QMessageBox.information(
                self,
                "링크 선택 필요",
                "왼쪽 링크 트리에서 대상 링크를 선택하세요.",
            )
            return
        self.project.assign_parts(identifiers, target)
        self._after_topology_change(target)

    def unassign_selection_from_current_link(self) -> None:
        if self.project is None or self.current_link not in self.project.links:
            return
        identifiers = [
            str(item.data(ROLE_ID))
            for item in self.link_parts_list.selectedItems()
            if item.data(ROLE_ID) is not None
        ]
        identifiers = [
            part_id
            for part_id in dict.fromkeys(identifiers)
            if (
                part_id in self.project.parts
                and self.project.parts[part_id].link_name == self.current_link
            )
        ]
        if not identifiers:
            QMessageBox.information(
                self,
                "형상 선택 필요",
                "현재 링크 형상 목록에서 뺄 형상을 선택하세요.",
            )
            return
        target = self.current_link
        self.project.assign_parts(identifiers, None)
        self._after_topology_change(target)
        # Removed parts are now unassigned and therefore visible even while the
        # assigned-parts filter is active. Keep them selected as clear feedback.
        self._set_selected_parts(identifiers)
        self.statusBar().showMessage(
            f"{target}: 형상 {len(identifiers)}개를 미할당으로 뺐습니다.",
            10000,
        )

    def assign_selection_to_base(self) -> None:
        if self.project is None or self.project.root_link is None:
            return
        identifiers = self._selected_part_ids()
        if identifiers:
            self.project.assign_parts(identifiers, self.project.root_link)
            self._after_topology_change(self.project.root_link)

    def delete_current_link(self) -> None:
        if self.project is None or not self.current_link:
            return
        if self.current_link == self.project.root_link:
            QMessageBox.information(
                self,
                "Base 링크는 삭제할 수 없음",
                "Base를 다시 만들려면 '전체 새로 시작'을 사용하세요.",
            )
            return
        incoming = self.project.joint_for_child(self.current_link)
        if incoming is None:
            return
        child_count = len(self.project.children_of(self.current_link))
        child_message = (
            f"\n하위 링크 {child_count}개는 {incoming.parent} 아래에 그대로 유지됩니다."
            if child_count
            else ""
        )
        if QMessageBox.question(
            self,
            "링크 삭제",
            (
                f"{self.current_link} 링크를 삭제할까요?\n"
                "이 링크의 형상은 미할당 상태가 됩니다."
                f"{child_message}"
            ),
        ) != QMessageBox.StandardButton.Yes:
            return
        parent = incoming.parent
        unassigned = list(self.project.links[self.current_link].part_ids)
        self.project.assign_parts(unassigned, None)
        self.project.merge_links(parent, self.current_link)
        self.current_link = parent
        parent_joint = self.project.joint_for_child(parent)
        self.current_joint = parent_joint.name if parent_joint else None
        self._after_topology_change(parent)
        self._set_selected_parts(unassigned)
        self.statusBar().showMessage(
            f"링크를 삭제하고 형상 {len(unassigned)}개를 미할당 상태로 돌렸습니다.",
            10000,
        )

    # Backward-compatible name retained for any older integrations.
    def merge_current_link(self) -> None:
        self.delete_current_link()

    def _after_topology_change(self, link_name: str) -> None:
        self._prune_mechanism_metadata()
        self.current_link = link_name
        incoming = self.project.joint_for_child(link_name) if self.project else None
        self.current_joint = incoming.name if incoming else None
        self._mark_dirty()
        self._rebuild_all(preserve_selection=False)
        self._select_link_item(link_name)
        self._schedule_collision_precheck()

    def _prune_mechanism_metadata(self) -> None:
        if self.project is None:
            return
        mechanisms = self.project.metadata.get("mechanisms")
        if not isinstance(mechanisms, list):
            return
        valid_links = set(self.project.links)
        valid_joints = {joint.name for joint in self.project.joints}
        mechanisms[:] = [
            item
            for item in mechanisms
            if isinstance(item, dict)
            and item.get("link") in valid_links
            and item.get("joint") in valid_joints
        ]

    # ---------- joint editing and preview ----------
    def apply_joint_values(self, values: dict[str, Any]) -> None:
        if self.project is None or self.current_joint is None:
            return
        try:
            index = next(i for i, item in enumerate(self.project.joints) if item.name == self.current_joint)
        except StopIteration:
            return
        old_joint = self.project.joints[index]
        name = sanitize_name(values["name"], "joint")
        if any(joint.name == name and joint is not old_joint for joint in self.project.joints):
            QMessageBox.warning(self, "관절 이름 중복", name)
            return
        replacement = copy.deepcopy(old_joint)
        replacement.name = name
        replacement.type = values["type"]
        replacement.parent = values["parent"]
        replacement.child = values["child"]
        replacement.origin_xyz = np.asarray(values["origin_xyz"], dtype=float)
        replacement.origin_rpy = np.asarray(values["origin_rpy"], dtype=float)
        replacement.axis = np.asarray(values["axis"], dtype=float)
        replacement.lower = None if replacement.type == "continuous" else float(values["lower"])
        replacement.upper = None if replacement.type == "continuous" else float(values["upper"])
        replacement.position = replacement.clamp(float(values["position"]))
        replacement.effort = float(values.get("effort", replacement.effort))
        replacement.velocity = float(values.get("velocity", replacement.velocity))
        replacement.damping = float(values.get("damping", replacement.damping))
        replacement.friction = float(values.get("friction", replacement.friction))
        replacement.mimic_joint = values.get("mimic_joint")
        replacement.mimic_auto = bool(values.get("mimic_auto", False))
        replacement.mimic_reverse = bool(values.get("mimic_reverse", False))
        replacement.mimic_multiplier = float(values.get("mimic_multiplier", 1.0))
        replacement.mimic_offset = float(values.get("mimic_offset", 0.0))
        replacement.drive_source_joint = values.get("drive_source_joint")
        replacement.drive_max_velocity = float(
            values.get("drive_max_velocity", 2.0 * math.pi)
        )
        replacement.drive_deadband = float(values.get("drive_deadband", 0.03))
        replacement.drive_reverse = bool(values.get("drive_reverse", False))
        if replacement.type == "fixed":
            replacement.lower = replacement.upper = replacement.position = 0.0
            replacement.mimic_joint = None
            replacement.mimic_auto = False
            replacement.mimic_reverse = False
        if replacement.type != "continuous":
            replacement.drive_source_joint = None
            replacement.drive_reverse = False
        renamed_dependents = [
            joint
            for joint in self.project.joints
            if joint is not old_joint and joint.mimic_joint == old_joint.name
        ]
        renamed_drive_targets = [
            joint
            for joint in self.project.joints
            if joint is not old_joint
            and joint.drive_source_joint == old_joint.name
        ]
        self.project.joints[index] = replacement
        if name != old_joint.name:
            for dependent in renamed_dependents:
                dependent.mimic_joint = name
            for target in renamed_drive_targets:
                target.drive_source_joint = name
        errors = self.project.validate(check_names=False)
        if errors:
            self.project.joints[index] = old_joint
            for dependent in renamed_dependents:
                dependent.mimic_joint = old_joint.name
            for target in renamed_drive_targets:
                target.drive_source_joint = old_joint.name
            QMessageBox.warning(self, "관절 설정 오류", "\n".join(errors))
            return
        if name != old_joint.name:
            mechanisms = self.project.metadata.get("mechanisms")
            if isinstance(mechanisms, list):
                for item in mechanisms:
                    if not isinstance(item, dict):
                        continue
                    if item.get("joint") == old_joint.name:
                        item["joint"] = name
                    if item.get("source_joint") == old_joint.name:
                        item["source_joint"] = name
        self.project.apply_mimic_positions()
        self.current_joint = replacement.name
        self.current_link = replacement.child
        self._mark_dirty()
        self._rebuild_link_tree()
        self._refresh_editor()
        self._update_scene_transforms()
        self._schedule_collision_precheck()

    def set_current_joint_position(self, value: float) -> None:
        if self.project is None or self.current_joint is None:
            return
        try:
            self.project.set_joint_position(self.current_joint, value)
        except (KeyError, ValueError):
            return
        self._update_scene_transforms()
        self._mark_dirty()

    def use_selection_center_for_origin(self) -> None:
        if self.project is None or self.current_joint is None:
            return
        identifiers = self._selected_part_ids()
        if not identifiers:
            QMessageBox.information(self, "형상 선택 필요", "원점으로 사용할 형상을 선택하세요.")
            return
        joint = self.project.joint(self.current_joint)
        try:
            parent_zero = self.project.forward_kinematics(zero=True)[joint.parent]
        except Exception as exc:
            QMessageBox.warning(self, "원점 설정 실패", str(exc))
            return
        center_parent = apply_transform(
            self._parts_center(identifiers).reshape(1, 3), np.linalg.inv(parent_zero)
        )[0]
        self.joint_editor.set_origin_mm(center_parent * 1000.0)

    def preview_joint_axis(self, axis: Iterable[float]) -> None:
        """Preview an editor axis shortcut or direction flip before applying."""

        if self.project is None or self.current_joint is None:
            return
        try:
            joint = self.project.joint(self.current_joint)
            axis_joint = np.asarray(axis, dtype=float)
            axis_joint /= np.linalg.norm(axis_joint)
            current_fk = self.project.forward_kinematics()
            parent_frame = current_fk[joint.parent]
            origin_parent = self.joint_editor.origin_editor.value() / 1000.0
            origin_world = apply_transform(
                origin_parent.reshape(1, 3),
                parent_frame,
            )[0]
            direction_world = (
                parent_frame[:3, :3]
                @ rpy_matrix(joint.origin_rpy)
                @ axis_joint
            )
            rotational = self.joint_editor.type_combo.currentText() in {
                "revolute",
                "continuous",
            }
            if not rotational:
                marker_origin = self._current_link_bbox_center(current_fk)
                if marker_origin is None:
                    marker_origin = origin_world
            else:
                marker_origin = origin_world
            self.viewport.set_axis_marker(
                marker_origin,
                direction_world,
                self._scene_scale * (0.20 if rotational else 0.10),
                bidirectional=rotational,
            )
        except (KeyError, ValueError, np.linalg.LinAlgError):
            return

    # ---------- state ----------
    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            self._update_title()

    def _update_title(self) -> None:
        name = self.project.name if self.project else "STEP URDF Maker"
        suffix = " *" if self._dirty else ""
        self.setWindowTitle(f"{name} — STEP URDF Maker{suffix}")

    def _set_actions_enabled(self, enabled: bool) -> None:
        for action in (
            self.save_project_action,
            self.save_as_action,
            self.export_action,
            self.frame_action,
            self.clear_selection_action,
            self.part_colors_action,
            self.hide_assigned_action,
            self.mechanism_wizard_action,
        ):
            action.setEnabled(enabled)
        for widget in (
            self.create_link_button,
            self.assign_button,
            self.to_base_button,
            self.new_tree_button,
            self.add_child_button,
            self.mechanism_wizard_button,
            self.assign_current_tree_button,
            self.unassign_current_tree_button,
            self.assigned_visibility_check,
            self.merge_button,
        ):
            widget.setEnabled(enabled)

    def _confirm_discard_changes(self) -> bool:
        if not self._dirty:
            return True
        answer = QMessageBox.question(
            self,
            "저장하지 않은 변경",
            "현재 변경 내용을 저장할까요?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            return self.save_project()
        return answer == QMessageBox.StandardButton.Discard

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._stop_demo_animation(restore=True)
        if self._new_link_dialog is not None:
            self._new_link_dialog.reject()
        if self._workers or self._progress is not None:
            QMessageBox.information(
                self,
                "작업 진행 중",
                "STEP/URDF 로딩 또는 메시 충돌 검사가 끝난 뒤 종료해 주세요. "
                "네이티브 작업을 안전하게 정리하고 있습니다.",
            )
            event.ignore()
            return
        # Closing the application is immediate. Saving remains an explicit
        # Ctrl+S/File-menu action; repeated Save/Discard prompts made ordinary
        # preview and test sessions unnecessarily intrusive.
        event.accept()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        urls = event.mimeData().urls()
        if any(
            url.isLocalFile()
            and Path(url.toLocalFile()).suffix.lower() in {".step", ".stp", ".urdf", ".xml", ".json"}
            for url in urls
        ):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.suffix.lower() in {".step", ".stp", ".urdf", ".xml", ".json"}:
                self.open_path(path)
                event.acceptProposedAction()
                return


__all__ = ["MainWindow"]
