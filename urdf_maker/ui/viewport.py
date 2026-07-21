"""Interactive VTK viewport used by the STEP/URDF editor.

The widget deliberately owns only rendering and part selection.  Geometry is
supplied in the robot's zero-pose coordinate system and joint previews are
applied as per-part delta transforms, so callers do not need to rebuild meshes
while a slider is moving.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np

from ..runtime_env import preload_system_opengl


# Keep direct imports of this reusable widget hardware-accelerated too; the
# normal desktop entry point performs the same preload even earlier.
preload_system_opengl()


_IMPORT_ERROR: Exception | None = None

try:
    from PySide6.QtCore import (
        QEvent,
        QObject,
        QPointF,
        Qt,
        QTimer,
        Signal,
    )
    from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget

    # Importing these modules also registers the OpenGL render-window and the
    # default interaction style with VTK's object factory.
    import vtkmodules.vtkInteractionStyle  # noqa: F401
    import vtkmodules.vtkRenderingFreeType  # noqa: F401
    import vtkmodules.vtkRenderingOpenGL2  # noqa: F401
    from vtkmodules.qt.QVTKRenderWindowInteractor import (
        QVTKRenderWindowInteractor,
    )
    from vtkmodules.util.numpy_support import (
        numpy_to_vtk,
        numpy_to_vtkIdTypeArray,
    )
    from vtkmodules.vtkCommonCore import vtkCommand, vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkFiltersCore import vtkPolyDataNormals
    from vtkmodules.vtkFiltersHybrid import vtkPolyDataSilhouette
    from vtkmodules.vtkFiltersSources import vtkArrowSource, vtkLineSource
    from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
    from vtkmodules.vtkInteractionWidgets import vtkOrientationMarkerWidget
    from vtkmodules.vtkRenderingAnnotation import vtkAxesActor
    from vtkmodules.vtkRenderingCore import (
        vtkActor,
        vtkCellPicker,
        vtkPolyDataMapper,
        vtkRenderer,
    )
except Exception as exc:  # pragma: no cover - exercised on dependency failures
    _IMPORT_ERROR = exc


__all__ = [
    "ViewportWidget",
    "VTKViewport",
    "VTKViewportWidget",
    "Viewport3D",
]


def _missing_dependency_message() -> str:
    detail = f" ({_IMPORT_ERROR})" if _IMPORT_ERROR is not None else ""
    return (
        "The 3D viewport requires PySide6 and VTK with Qt support. "
        "Install the project's UI dependencies (for example: "
        "`pip install PySide6 vtk numpy`) and restart the application."
        f"{detail}"
    )


if _IMPORT_ERROR is not None:

    class ViewportWidget:  # type: ignore[no-redef]
        """Dependency-error placeholder which keeps this module importable."""

        partsSelectionChanged = None

        @staticmethod
        def dependency_error() -> str:
            return _missing_dependency_message()

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(_missing_dependency_message()) from _IMPORT_ERROR


else:

    _MISSING = object()
    _DEFAULT_COLOR = (0.70, 0.74, 0.80, 1.0)
    _MAX_RENDER_FPS = 60.0
    _FRAME_INTERVAL_SECONDS = 1.0 / _MAX_RENDER_FPS


    @dataclass(slots=True)
    class _PartActor:
        part_id: str
        name: str
        actor: vtkActor
        polydata: vtkPolyData
        normals: vtkPolyDataNormals
        mapper: vtkPolyDataMapper
        color: tuple[float, float, float, float]


    @dataclass(slots=True)
    class _SelectionOutline:
        silhouette: vtkPolyDataSilhouette
        mapper: vtkPolyDataMapper
        actor: vtkActor


    def _part_field(part: Any, name: str, default: Any = _MISSING) -> Any:
        if isinstance(part, Mapping):
            if name in part:
                return part[name]
        elif hasattr(part, name):
            return getattr(part, name)

        if default is not _MISSING:
            return default
        raise ValueError(f"Part is missing the required '{name}' field")


    def _coerce_color(value: Any) -> tuple[float, float, float, float]:
        if value is None:
            return _DEFAULT_COLOR

        # QColor and compatible classes expose getRgbF().
        if hasattr(value, "getRgbF"):
            rgba = np.asarray(value.getRgbF(), dtype=np.float64)
        else:
            try:
                rgba = np.asarray(value, dtype=np.float64).reshape(-1)
            except (TypeError, ValueError) as exc:
                raise ValueError("Part color must contain RGB or RGBA values") from exc

        if rgba.size not in (3, 4) or not np.all(np.isfinite(rgba)):
            raise ValueError("Part color must contain three or four finite values")
        rgba = rgba.copy()
        if rgba.size == 3:
            rgba = np.concatenate((rgba, np.asarray((1.0,))))

        # Accept both the common 0..1 representation and 8-bit colors.  RGB
        # and alpha are handled separately because ``(255, 0, 0, 1.0)`` is a
        # frequently produced mixed representation.
        if np.max(rgba[:3]) > 1.0:
            rgba[:3] = rgba[:3] / 255.0
        if rgba[3] > 1.0:
            rgba[3] = rgba[3] / 255.0
        rgba = np.clip(rgba, 0.0, 1.0)
        return tuple(float(component) for component in rgba)  # type: ignore[return-value]


    def _coerce_vertices(value: Any, part_id: str) -> np.ndarray:
        try:
            vertices = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Part '{part_id}' has invalid vertices_zero data") from exc

        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError(
                f"Part '{part_id}' vertices_zero must have shape (N, 3), "
                f"got {vertices.shape}"
            )
        if not np.all(np.isfinite(vertices)):
            raise ValueError(f"Part '{part_id}' vertices_zero contains non-finite values")
        return np.ascontiguousarray(vertices, dtype=np.float64)


    def _coerce_triangles(value: Any, vertex_count: int, part_id: str) -> np.ndarray:
        raw = np.asarray(value)
        if raw.ndim == 1 and raw.size % 3 == 0:
            raw = raw.reshape((-1, 3))
        if raw.ndim != 2 or raw.shape[1:] != (3,):
            raise ValueError(
                f"Part '{part_id}' triangles must have shape (M, 3), got {raw.shape}"
            )

        if not np.issubdtype(raw.dtype, np.integer):
            try:
                numeric = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Part '{part_id}' has invalid triangle indices") from exc
            if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.rint(numeric)):
                raise ValueError(f"Part '{part_id}' triangle indices must be integers")
            raw = np.rint(numeric)

        triangles = np.ascontiguousarray(raw, dtype=np.int64)
        if triangles.size:
            minimum = int(np.min(triangles))
            maximum = int(np.max(triangles))
            if minimum < 0 or maximum >= vertex_count:
                raise ValueError(
                    f"Part '{part_id}' triangle index range [{minimum}, {maximum}] "
                    f"is outside its {vertex_count} vertices"
                )
        return triangles


    def _matrix4(value: Any, *, label: str) -> np.ndarray:
        try:
            matrix = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a numeric 4x4 matrix") from exc
        if matrix.shape != (4, 4):
            raise ValueError(f"{label} must have shape (4, 4), got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{label} contains non-finite values")
        return matrix


    def _vtk_matrix(matrix: np.ndarray) -> vtkMatrix4x4:
        result = vtkMatrix4x4()
        for row in range(4):
            for column in range(4):
                result.SetElement(row, column, float(matrix[row, column]))
        return result


    def _actor_key(actor: vtkActor | None) -> str | None:
        if actor is None:
            return None
        return actor.GetAddressAsString("")


    class ViewportWidget(QWidget):  # type: ignore[no-redef]
        """A reusable STEP/URDF mesh viewport with click selection.

        ``vertices_zero`` is expected to contain zero-pose world coordinates.
        ``update_part_transforms`` therefore accepts each part's FK delta
        ``T(q) @ inverse(T(0))`` and applies it directly to the corresponding
        actor.
        """

        partsSelectionChanged = Signal(list)
        animationToggled = Signal(bool)

        def __init__(self, parent: QWidget | None = None) -> None:
            super().__init__(parent)

            self.setObjectName("modelViewport")
            self.setMinimumSize(240, 180)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            self._vtk_widget = QVTKRenderWindowInteractor(self)
            self._vtk_widget.setObjectName("vtkRenderWindow")
            self._vtk_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            layout.addWidget(self._vtk_widget)

            self._renderer = vtkRenderer()
            self._renderer.SetBackground(0.065, 0.075, 0.095)
            self._renderer.SetBackground2(0.18, 0.20, 0.24)
            self._renderer.GradientBackgroundOn()

            self._render_window = self._vtk_widget.GetRenderWindow()
            self._render_window.AddRenderer(self._renderer)
            self._render_window.SetMultiSamples(4)
            self._render_window.SetDesiredUpdateRate(_MAX_RENDER_FPS)
            if hasattr(self._render_window, "SetSwapControl"):
                self._render_window.SetSwapControl(1)

            self._interactor = self._render_window.GetInteractor()
            self._interaction_style = vtkInteractorStyleTrackballCamera()
            self._interactor.SetInteractorStyle(self._interaction_style)
            self._interactor.SetDesiredUpdateRate(_MAX_RENDER_FPS)

            self._last_render_started = 0.0
            self._render_pending = False
            self._render_timer = QTimer(self)
            self._render_timer.setSingleShot(True)
            self._render_timer.timeout.connect(self._perform_scheduled_render)
            self._resize_render_timer = QTimer(self)
            self._resize_render_timer.setSingleShot(True)
            self._resize_render_timer.timeout.connect(self._render_after_resize)

            self._frames_since_sample = 0
            self._fps_sample_started = time.perf_counter()
            self._last_render_completed = 0.0
            self._fps_label = QLabel("FPS: 대기", self._vtk_widget)
            self._fps_label.setObjectName("viewportFps")
            self._fps_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self._fps_label.setStyleSheet(
                "QLabel#viewportFps {"
                " color: #f2f4f8;"
                " background: rgba(18, 22, 30, 180);"
                " border: 1px solid rgba(210, 218, 230, 90);"
                " border-radius: 4px;"
                " padding: 3px 7px;"
                " font-family: Consolas, monospace;"
                " font-size: 11px;"
                "}"
            )
            self._fps_label.adjustSize()
            self._position_fps_label()
            self._fps_label.raise_()
            self._shortcut_label = QLabel(
                "Ctrl+H: 배정 파츠 숨김/보임 · F: 전체 보기 · Esc: 선택 해제",
                self._vtk_widget,
            )
            self._shortcut_label.setObjectName("viewportShortcuts")
            self._shortcut_label.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents
            )
            self._shortcut_label.setStyleSheet(
                "QLabel#viewportShortcuts {"
                " color: #e5e9f0;"
                " background: rgba(18, 22, 30, 170);"
                " border: 1px solid rgba(210, 218, 230, 70);"
                " border-radius: 4px;"
                " padding: 3px 7px;"
                " font-size: 11px;"
                "}"
            )
            self._shortcut_label.adjustSize()
            self._position_shortcut_label()
            self._shortcut_label.raise_()
            self._play_button = QPushButton("▶ Play", self._vtk_widget)
            self._play_button.setObjectName("viewportPlayButton")
            self._play_button.setCheckable(True)
            self._play_button.setToolTip(
                "카메라를 천천히 회전하고 모든 가동 관절을 한계 사이에서 왕복합니다."
            )
            self._play_button.setStyleSheet(
                "QPushButton#viewportPlayButton {"
                " color: #f5f7fa;"
                " background: rgba(34, 42, 54, 220);"
                " border: 1px solid rgba(210, 218, 230, 120);"
                " border-radius: 5px;"
                " padding: 4px 10px;"
                " font-weight: 600;"
                "}"
                "QPushButton#viewportPlayButton:hover {"
                " background: rgba(52, 66, 84, 235);"
                "}"
                "QPushButton#viewportPlayButton:checked {"
                " background: rgba(156, 48, 38, 235);"
                "}"
            )
            self._play_button.toggled.connect(self._play_toggled)
            self._play_button.adjustSize()
            self._position_play_button()
            self._play_button.raise_()
            self._fps_timer = QTimer(self)
            self._fps_timer.setInterval(500)
            self._fps_timer.timeout.connect(self._refresh_fps_label)
            self._fps_timer.start()
            self._render_window.AddObserver(vtkCommand.EndEvent, self._render_completed)

            self._picker = vtkCellPicker()
            self._picker.SetTolerance(0.0008)
            self._picker.PickFromListOn()

            self._parts: dict[str, _PartActor] = {}
            self._actor_to_part: dict[str, str] = {}
            self._color_overrides: dict[str, tuple[float, float, float, float]] = {}
            self._selected: list[str] = []
            self._mouse_press_position: QPointF | None = None
            self._mouse_press_ctrl = False
            self._mouse_press_remove = False
            self._has_framed = False

            self._axis_actor: vtkActor | None = None
            self._axis_source: vtkArrowSource | vtkLineSource | None = None
            self._axis_bidirectional = False

            # Give both the main renderer and the orientation marker a valid
            # non-polar camera from their first render. Setting Z-up while the
            # default camera is looking exactly along Z makes VTK emit a noisy
            # "view-up is parallel" warning before the first model is framed.
            camera = self._renderer.GetActiveCamera()
            camera.SetFocalPoint(0.0, 0.0, 0.0)
            camera.SetPosition(1.0, -1.0, 0.72)
            camera.SetViewUp(0.0, 0.0, 1.0)
            camera.SetClippingRange(0.001, 1000.0)

            self._orientation_axes = self._make_orientation_axes()
            self._orientation_widget = vtkOrientationMarkerWidget()
            self._orientation_widget.SetOrientationMarker(self._orientation_axes)
            self._orientation_widget.SetInteractor(self._interactor)
            self._orientation_widget.SetViewport(0.0, 0.0, 0.16, 0.16)
            self._orientation_widget.SetOutlineColor(0.72, 0.75, 0.80)
            self._orientation_widget.SetEnabled(1)
            self._orientation_widget.SetInteractive(0)

            # Selection outlines render in their own full-viewport layer. Its
            # fresh depth buffer makes the complete outline visible through
            # intervening geometry, while the orientation marker stays above
            # it in a final small overlay layer.
            self._render_window.SetNumberOfLayers(3)
            orientation_renderer = self._orientation_widget.GetRenderer()
            if orientation_renderer is not None:
                orientation_renderer.SetLayer(2)
            self._selection_renderer = vtkRenderer()
            self._selection_renderer.SetLayer(1)
            self._selection_renderer.SetInteractive(False)
            self._selection_renderer.SetPreserveDepthBuffer(False)
            self._selection_renderer.SetActiveCamera(camera)
            self._render_window.AddRenderer(self._selection_renderer)
            self._selection_outlines: dict[str, _SelectionOutline] = {}

            self._vtk_widget.installEventFilter(self)

            # VTK interaction styles normally render immediately on every
            # mouse event. Disable that path and funnel interaction renders
            # through the same 60 Hz scheduler used by the rest of the app.
            self._interactor.EnableRenderOff()
            self._interaction_style.AddObserver(
                vtkCommand.InteractionEvent,
                lambda _caller, _event: self._render(),
            )
            self._interaction_style.AddObserver(
                vtkCommand.EndInteractionEvent,
                lambda _caller, _event: self._render(),
            )

            # QVTKRenderWindowInteractor.Initialize() is safe to call more than
            # once and avoids a blank first frame on some Windows/Qt builds.
            self._interactor.Initialize()

        @staticmethod
        def dependency_error() -> str | None:
            """Return a dependency diagnostic; ``None`` for a working widget."""

            return None

        @property
        def renderer(self) -> vtkRenderer:
            """Expose the renderer for read-only integration/debug tooling."""

            return self._renderer

        @property
        def interactor(self) -> Any:
            """Return VTK's render-window interactor."""

            return self._interactor

        def set_parts(self, parts: Iterable[Any]) -> None:
            """Replace the displayed parts.

            Each part may be an object or mapping and must provide ``id``,
            ``name``, ``vertices_zero``, ``triangles``, ``color`` and
            ``visible``. ``name``, ``color`` and ``visible`` have sensible
            defaults. ``vertices`` is accepted as a compatibility alias for
            STEP loaders while they expose ``vertices_zero`` as a property.

            Validation and VTK actor construction happen before the current
            scene is changed, preventing a malformed part from leaving a
            half-populated viewport.
            """

            built: list[_PartActor] = []
            seen: set[str] = set()

            for part in parts:
                raw_id = _part_field(part, "id")
                if raw_id is None or str(raw_id) == "":
                    raise ValueError("Every part must have a non-empty id")
                part_id = str(raw_id)
                if part_id in seen:
                    raise ValueError(f"Duplicate part id: '{part_id}'")
                seen.add(part_id)

                name = str(_part_field(part, "name", part_id))
                try:
                    vertices_value = _part_field(part, "vertices_zero")
                except ValueError:
                    vertices_value = _part_field(part, "vertices")
                vertices = _coerce_vertices(vertices_value, part_id)
                triangles = _coerce_triangles(
                    _part_field(part, "triangles"), len(vertices), part_id
                )
                color = _coerce_color(_part_field(part, "color", None))
                visible = bool(_part_field(part, "visible", True))

                actor_data = self._build_actor(
                    part_id, name, vertices, triangles, color
                )
                actor_data.actor.SetVisibility(visible)
                built.append(actor_data)

            previous_selection = list(self._selected)
            self._remove_parts()

            for record in built:
                self._parts[record.part_id] = record
                self._actor_to_part[_actor_key(record.actor) or ""] = record.part_id
                self._renderer.AddActor(record.actor)
                self._picker.AddPickList(record.actor)

            self._selected = [
                part_id for part_id in previous_selection if part_id in self._parts
            ]
            self._color_overrides = {
                part_id: color
                for part_id, color in self._color_overrides.items()
                if part_id in self._parts
            }
            self._apply_selection_style()
            self._has_framed = False

            if self._visible_bounds() is not None:
                self.frame_all()
            else:
                self._render()

            if self._selected != previous_selection:
                self.partsSelectionChanged.emit(list(self._selected))

        def update_part_transforms(self, transforms: Mapping[str, Any]) -> None:
            """Apply 4x4 zero-pose delta transforms without moving the camera.

            Unknown ids are ignored so callers may pass a complete FK result
            containing links which have no rendered geometry.
            """

            changed = False
            for raw_part_id, value in transforms.items():
                part_id = str(raw_part_id)
                record = self._parts.get(part_id)
                if record is None:
                    continue
                matrix = _matrix4(value, label=f"Transform for part '{part_id}'")
                vtk_matrix = _vtk_matrix(matrix)
                record.actor.SetUserMatrix(vtk_matrix)
                outline = self._selection_outlines.get(part_id)
                if outline is not None:
                    outline.actor.SetUserMatrix(vtk_matrix)
                changed = True

            if changed:
                self._renderer.ResetCameraClippingRange()
                self._render()

        def set_selected(self, ids: Iterable[str]) -> None:
            """Select known part ids, preserving caller order and uniqueness."""

            selected: list[str] = []
            seen: set[str] = set()
            for raw_id in ids:
                part_id = str(raw_id)
                if part_id in self._parts and part_id not in seen:
                    selected.append(part_id)
                    seen.add(part_id)
            self._set_selection(selected)

        def selected_ids(self) -> list[str]:
            """Return a copy of the selected part id list."""

            return list(self._selected)

        def set_color_overrides(self, colors: Mapping[str, Any]) -> None:
            """Set transient display colors without changing CAD/URDF materials."""

            self._color_overrides = {
                str(part_id): _coerce_color(color)
                for part_id, color in colors.items()
                if str(part_id) in self._parts
            }
            self._apply_selection_style()
            self._render()

        def frame_all(self) -> None:
            """Fit all visible part actors while keeping a stable view axis."""

            bounds = self._visible_bounds()
            if bounds is None:
                return

            lower = np.asarray((bounds[0], bounds[2], bounds[4]), dtype=np.float64)
            upper = np.asarray((bounds[1], bounds[3], bounds[5]), dtype=np.float64)
            center = (lower + upper) * 0.5
            radius = max(float(np.linalg.norm(upper - lower) * 0.5), 1.0e-4)

            camera = self._renderer.GetActiveCamera()
            if not self._has_framed:
                view_direction = np.asarray((1.0, -1.0, 0.72), dtype=np.float64)
                view_direction /= np.linalg.norm(view_direction)
                view_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            else:
                position = np.asarray(camera.GetPosition(), dtype=np.float64)
                focal = np.asarray(camera.GetFocalPoint(), dtype=np.float64)
                view_direction = position - focal
                norm = float(np.linalg.norm(view_direction))
                if not math.isfinite(norm) or norm < 1.0e-9:
                    view_direction = np.asarray((1.0, -1.0, 0.72), dtype=np.float64)
                    view_direction /= np.linalg.norm(view_direction)
                else:
                    view_direction /= norm
                view_up = np.asarray(camera.GetViewUp(), dtype=np.float64)
                if not np.all(np.isfinite(view_up)) or np.linalg.norm(view_up) < 1.0e-9:
                    view_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)

            # VTK warns and silently replaces the view-up vector when it is
            # parallel to the viewing direction.  Preserve the user's current
            # orientation where possible and choose a deterministic fallback
            # near the poles.
            view_up = view_up - np.dot(view_up, view_direction) * view_direction
            view_up_norm = float(np.linalg.norm(view_up))
            if not math.isfinite(view_up_norm) or view_up_norm < 1.0e-9:
                fallback = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
                if abs(float(np.dot(fallback, view_direction))) > 0.95:
                    fallback = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
                view_up = fallback - np.dot(fallback, view_direction) * view_direction
                view_up_norm = float(np.linalg.norm(view_up))
            view_up /= view_up_norm

            half_angle = math.radians(max(float(camera.GetViewAngle()), 1.0) * 0.5)
            distance = radius / max(math.sin(half_angle), 0.05) * 1.12
            camera.SetFocalPoint(*(float(value) for value in center))
            camera.SetPosition(
                *(float(value) for value in center + view_direction * distance)
            )
            camera.SetViewUp(*(float(value) for value in view_up))
            camera.OrthogonalizeViewUp()
            camera.SetParallelScale(radius * 1.15)

            self._renderer.ResetCameraClippingRange()
            self._has_framed = True
            self._render()

        def set_axis_marker(
            self,
            origin: Any,
            direction: Any,
            length: float,
            *,
            bidirectional: bool = False,
        ) -> None:
            """Show a joint direction arrow or a centered rotation-axis line."""

            origin_vector = np.asarray(origin, dtype=np.float64).reshape(-1)
            direction_vector = np.asarray(direction, dtype=np.float64).reshape(-1)
            if origin_vector.shape != (3,) or not np.all(np.isfinite(origin_vector)):
                raise ValueError("Axis origin must contain three finite values")
            if direction_vector.shape != (3,) or not np.all(
                np.isfinite(direction_vector)
            ):
                raise ValueError("Axis direction must contain three finite values")

            magnitude = float(np.linalg.norm(direction_vector))
            if magnitude < 1.0e-12:
                raise ValueError("Axis direction must be non-zero")
            try:
                marker_length = float(length)
            except (TypeError, ValueError) as exc:
                raise ValueError("Axis marker length must be a positive number") from exc
            if not math.isfinite(marker_length) or marker_length <= 0.0:
                raise ValueError("Axis marker length must be a positive finite number")

            direction_unit = direction_vector / magnitude
            helper = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
            if abs(float(np.dot(direction_unit, helper))) > 0.92:
                helper = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
            basis_y = np.cross(helper, direction_unit)
            basis_y /= np.linalg.norm(basis_y)
            basis_z = np.cross(direction_unit, basis_y)

            transform = np.eye(4, dtype=np.float64)
            transform[:3, 0] = direction_unit * marker_length
            transform[:3, 1] = basis_y * marker_length
            transform[:3, 2] = basis_z * marker_length
            transform[:3, 3] = origin_vector

            marker_is_bidirectional = bool(bidirectional)
            if (
                self._axis_actor is not None
                and self._axis_bidirectional != marker_is_bidirectional
            ):
                self._selection_renderer.RemoveActor(self._axis_actor)
                self._axis_actor = None
                self._axis_source = None

            if self._axis_actor is None:
                if marker_is_bidirectional:
                    line = vtkLineSource()
                    line.SetPoint1(-1.0, 0.0, 0.0)
                    line.SetPoint2(1.0, 0.0, 0.0)
                    self._axis_source = line
                else:
                    arrow = vtkArrowSource()
                    arrow.SetShaftResolution(24)
                    arrow.SetTipResolution(32)
                    arrow.SetShaftRadius(0.022)
                    arrow.SetTipRadius(0.070)
                    arrow.SetTipLength(0.28)
                    self._axis_source = arrow

                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(self._axis_source.GetOutputPort())
                self._axis_actor = vtkActor()
                self._axis_actor.SetMapper(mapper)
                self._axis_actor.SetPickable(False)
                self._axis_actor.GetProperty().SetColor(1.0, 0.20, 0.08)
                self._axis_actor.GetProperty().SetAmbient(0.35)
                self._axis_actor.GetProperty().SetDiffuse(0.75)
                if marker_is_bidirectional:
                    self._axis_actor.GetProperty().SetLineWidth(5.0)
                    self._axis_actor.GetProperty().SetRenderLinesAsTubes(True)
                self._selection_renderer.AddActor(self._axis_actor)
                self._axis_bidirectional = marker_is_bidirectional

            self._axis_actor.SetUserMatrix(_vtk_matrix(transform))
            self._axis_actor.SetVisibility(True)
            self._renderer.ResetCameraClippingRange()
            self._render()

        def clear_axis_marker(self) -> None:
            """Remove the world-space joint-axis preview, if present."""

            if self._axis_actor is not None:
                self._selection_renderer.RemoveActor(self._axis_actor)
                self._axis_actor = None
                self._axis_source = None
                self._axis_bidirectional = False
                self._renderer.ResetCameraClippingRange()
                self._render()

        def clear(self) -> None:
            """Remove all model actors, selection, and the joint-axis marker."""

            had_selection = bool(self._selected)
            self._remove_parts()
            if self._axis_actor is not None:
                self._selection_renderer.RemoveActor(self._axis_actor)
                self._axis_actor = None
                self._axis_source = None
                self._axis_bidirectional = False
            self._selected = []
            self._color_overrides.clear()
            self._has_framed = False
            self._render()
            if had_selection:
                self.partsSelectionChanged.emit([])

        def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
            """Turn a short left click into selection, leaving drags to VTK."""

            if watched is self._vtk_widget:
                event_type = event.type()
                if event_type == QEvent.Type.Resize:
                    self._position_fps_label()
                    self._position_shortcut_label()
                    self._position_play_button()
                    self._queue_resize_render()
                elif event_type == QEvent.Type.Show:
                    self._queue_resize_render()
                elif (
                    event_type == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton  # type: ignore[attr-defined]
                ):
                    self._mouse_press_position = QPointF(
                        event.position()  # type: ignore[attr-defined]
                    )
                    self._mouse_press_ctrl = bool(
                        event.modifiers()  # type: ignore[attr-defined]
                        & Qt.KeyboardModifier.ControlModifier
                    )
                    self._mouse_press_remove = bool(
                        event.modifiers()  # type: ignore[attr-defined]
                        & (
                            Qt.KeyboardModifier.ShiftModifier
                            | Qt.KeyboardModifier.AltModifier
                        )
                    )
                elif (
                    event_type == QEvent.Type.MouseButtonRelease
                    and event.button() == Qt.MouseButton.LeftButton  # type: ignore[attr-defined]
                    and self._mouse_press_position is not None
                ):
                    release_position = QPointF(event.position())  # type: ignore[attr-defined]
                    delta = release_position - self._mouse_press_position
                    click_distance = math.hypot(float(delta.x()), float(delta.y()))
                    drag_threshold = max(3.0, QApplication.startDragDistance() * 0.45)
                    control_pressed = self._mouse_press_ctrl
                    remove_pressed = self._mouse_press_remove
                    self._mouse_press_position = None
                    self._mouse_press_ctrl = False
                    self._mouse_press_remove = False
                    if click_distance <= drag_threshold:
                        self._pick(
                            release_position,
                            add=control_pressed,
                            remove=remove_pressed,
                        )

            return super().eventFilter(watched, event)

        def _build_actor(
            self,
            part_id: str,
            name: str,
            vertices: np.ndarray,
            triangles: np.ndarray,
            color: tuple[float, float, float, float],
        ) -> _PartActor:
            points = vtkPoints()
            points.SetData(numpy_to_vtk(vertices, deep=True))

            cells = vtkCellArray()
            if len(triangles):
                vtk_cells = np.empty((len(triangles), 4), dtype=np.int64)
                vtk_cells[:, 0] = 3
                vtk_cells[:, 1:] = triangles
                cells.SetCells(
                    len(triangles),
                    numpy_to_vtkIdTypeArray(vtk_cells.reshape(-1), deep=True),
                )

            polydata = vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(cells)

            normals = vtkPolyDataNormals()
            normals.SetInputData(polydata)
            normals.SetFeatureAngle(45.0)
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.SplittingOn()
            normals.ComputePointNormalsOn()
            normals.ComputeCellNormalsOff()

            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(normals.GetOutputPort())
            mapper.ScalarVisibilityOff()

            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.SetPickable(True)
            prop = actor.GetProperty()
            prop.SetColor(*color[:3])
            prop.SetOpacity(color[3])
            prop.SetInterpolationToPhong()
            prop.SetAmbient(0.16)
            prop.SetDiffuse(0.76)
            prop.SetSpecular(0.16)
            prop.SetSpecularPower(24.0)

            return _PartActor(
                part_id=part_id,
                name=name,
                actor=actor,
                polydata=polydata,
                normals=normals,
                mapper=mapper,
                color=color,
            )

        def _make_orientation_axes(self) -> vtkAxesActor:
            axes = vtkAxesActor()
            axes.SetShaftTypeToCylinder()
            axes.SetCylinderRadius(0.045)
            axes.SetConeRadius(0.32)
            axes.SetConeResolution(32)
            axes.SetTotalLength(1.0, 1.0, 1.0)
            axes.SetNormalizedShaftLength(0.72, 0.72, 0.72)
            axes.SetNormalizedTipLength(0.28, 0.28, 0.28)
            for caption in (
                axes.GetXAxisCaptionActor2D(),
                axes.GetYAxisCaptionActor2D(),
                axes.GetZAxisCaptionActor2D(),
            ):
                caption.GetCaptionTextProperty().SetColor(0.92, 0.94, 0.98)
                caption.GetCaptionTextProperty().SetBold(False)
            return axes

        def _remove_parts(self) -> None:
            self._remove_selection_outlines()
            for record in self._parts.values():
                self._renderer.RemoveActor(record.actor)
            self._parts.clear()
            self._actor_to_part.clear()
            self._picker.InitializePickList()

        def _set_selection(self, selected: list[str]) -> None:
            if selected == self._selected:
                return
            self._selected = selected
            self._apply_selection_style()
            self._render()
            self.partsSelectionChanged.emit(list(self._selected))

        def _apply_selection_style(self) -> None:
            for part_id, record in self._parts.items():
                prop = record.actor.GetProperty()
                display_color = self._color_overrides.get(part_id, record.color)
                base_rgb = np.asarray(display_color[:3], dtype=np.float64)
                prop.SetColor(*(float(value) for value in base_rgb))
                # Filled actors retain their exact material. Selection is a
                # separate silhouette actor, not triangle-edge highlighting.
                prop.SetEdgeVisibility(False)
                prop.SetLineWidth(1.0)
                prop.SetAmbient(0.16)
                prop.SetDiffuse(0.76)
                prop.SetSpecular(0.16)
                prop.SetOpacity(display_color[3])
            self._rebuild_selection_outlines()

        def _remove_selection_outlines(self) -> None:
            for outline in self._selection_outlines.values():
                self._selection_renderer.RemoveActor(outline.actor)
            self._selection_outlines.clear()

        def _rebuild_selection_outlines(self) -> None:
            self._remove_selection_outlines()
            camera = self._renderer.GetActiveCamera()
            for part_id in self._selected:
                record = self._parts.get(part_id)
                if record is None or not record.actor.GetVisibility():
                    continue
                silhouette = vtkPolyDataSilhouette()
                silhouette.SetInputData(record.polydata)
                silhouette.SetCamera(camera)
                silhouette.SetEnableFeatureAngle(True)
                silhouette.SetFeatureAngle(60.0)
                silhouette.SetBorderEdges(True)

                mapper = vtkPolyDataMapper()
                mapper.SetInputConnection(silhouette.GetOutputPort())
                mapper.ScalarVisibilityOff()

                actor = vtkActor()
                actor.SetMapper(mapper)
                actor.SetPickable(False)
                if record.actor.GetUserMatrix() is not None:
                    actor.SetUserMatrix(record.actor.GetUserMatrix())
                prop = actor.GetProperty()
                prop.SetColor(1.0, 0.78, 0.05)
                prop.SetOpacity(0.98)
                prop.SetAmbient(1.0)
                prop.SetDiffuse(0.0)
                prop.SetLineWidth(3.5)
                prop.SetRenderLinesAsTubes(True)

                self._selection_renderer.AddActor(actor)
                self._selection_outlines[part_id] = _SelectionOutline(
                    silhouette=silhouette,
                    mapper=mapper,
                    actor=actor,
                )

        def _pick(self, position: QPointF, *, add: bool, remove: bool) -> None:
            widget_width = max(self._vtk_widget.width(), 1)
            widget_height = max(self._vtk_widget.height(), 1)
            render_width, render_height = self._render_window.GetSize()
            scale_x = render_width / widget_width if render_width > 0 else 1.0
            scale_y = render_height / widget_height if render_height > 0 else 1.0
            x = int(round(float(position.x()) * scale_x))
            y = int(round((widget_height - 1.0 - float(position.y())) * scale_y))

            picked = bool(self._picker.Pick(x, y, 0.0, self._renderer))
            actor = self._picker.GetActor() if picked else None
            part_id = self._actor_to_part.get(_actor_key(actor) or "")

            if part_id is None:
                if not add and not remove:
                    self._set_selection([])
                return

            if remove:
                self._set_selection(
                    [selected_id for selected_id in self._selected if selected_id != part_id]
                )
            elif add:
                selected = list(self._selected)
                if part_id not in selected:
                    selected.append(part_id)
                self._set_selection(selected)
            else:
                self._set_selection([part_id])

        def _visible_bounds(self) -> tuple[float, float, float, float, float, float] | None:
            lower = np.asarray((np.inf, np.inf, np.inf), dtype=np.float64)
            upper = np.asarray((-np.inf, -np.inf, -np.inf), dtype=np.float64)
            found = False

            for record in self._parts.values():
                if not record.actor.GetVisibility():
                    continue
                bounds = np.asarray(record.actor.GetBounds(), dtype=np.float64)
                if bounds.shape != (6,) or not np.all(np.isfinite(bounds)):
                    continue
                actor_lower = np.asarray((bounds[0], bounds[2], bounds[4]))
                actor_upper = np.asarray((bounds[1], bounds[3], bounds[5]))
                if np.any(actor_upper < actor_lower):
                    continue
                lower = np.minimum(lower, actor_lower)
                upper = np.maximum(upper, actor_upper)
                found = True

            if not found:
                return None
            return (
                float(lower[0]),
                float(upper[0]),
                float(lower[1]),
                float(upper[1]),
                float(lower[2]),
                float(upper[2]),
            )

        def _position_fps_label(self) -> None:
            self._fps_label.adjustSize()
            x = max(8, self._vtk_widget.width() - self._fps_label.width() - 10)
            self._fps_label.move(x, 8)

        def _position_shortcut_label(self) -> None:
            self._shortcut_label.adjustSize()
            play_width = self._play_button.width() if hasattr(self, "_play_button") else 0
            available_width = max(1, self._vtk_widget.width() - play_width - 24)
            x = max(
                8,
                (available_width - self._shortcut_label.width()) // 2,
            )
            y = max(8, self._vtk_widget.height() - self._shortcut_label.height() - 10)
            self._shortcut_label.move(x, y)

        def _position_play_button(self) -> None:
            self._play_button.adjustSize()
            x = max(8, self._vtk_widget.width() - self._play_button.width() - 10)
            y = max(8, self._vtk_widget.height() - self._play_button.height() - 8)
            self._play_button.move(x, y)

        def _play_toggled(self, playing: bool) -> None:
            self._play_button.setText("■ Stop" if playing else "▶ Play")
            self._play_button.adjustSize()
            self._position_play_button()
            self._position_shortcut_label()
            self._play_button.raise_()
            self.animationToggled.emit(bool(playing))

        def set_animation_playing(self, playing: bool) -> None:
            """Synchronize the overlay button without emitting a new request."""

            self._play_button.blockSignals(True)
            self._play_button.setChecked(bool(playing))
            self._play_button.blockSignals(False)
            self._play_button.setText("■ Stop" if playing else "▶ Play")
            self._play_button.adjustSize()
            self._position_play_button()
            self._position_shortcut_label()

        def _render_completed(self, _caller: Any, _event: str) -> None:
            self._frames_since_sample += 1
            self._last_render_completed = time.perf_counter()

        def _refresh_fps_label(self) -> None:
            now = time.perf_counter()
            elapsed = max(now - self._fps_sample_started, 1.0e-9)
            frame_count = self._frames_since_sample
            self._frames_since_sample = 0
            self._fps_sample_started = now
            if frame_count:
                fps = min(_MAX_RENDER_FPS, frame_count / elapsed)
                self._fps_label.setText(f"FPS: {fps:4.1f}")
            elif now - self._last_render_completed >= 0.75:
                # A static scene does not continuously render. Calling this
                # state 0 FPS suggests a renderer failure, so label it plainly.
                self._fps_label.setText("FPS: 대기")
            self._position_fps_label()
            self._fps_label.raise_()
            self._shortcut_label.raise_()
            self._play_button.raise_()

        def _queue_resize_render(self) -> None:
            """Coalesce layout changes and redraw after QVTK has its new size."""

            self._vtk_widget.update()
            self._resize_render_timer.start(0)

        def _render_after_resize(self) -> None:
            # QVTK's resizeEvent updates the native surface before this zero-ms
            # timer runs. An explicit VTK render is still required on Windows;
            # a Qt paint request alone can leave pixels from the adjacent dock.
            self._vtk_widget.update()
            self._render()

        def _perform_scheduled_render(self) -> None:
            self._render_pending = False
            if not self._vtk_widget.isVisible():
                self._vtk_widget.update()
                return
            self._last_render_started = time.perf_counter()
            self._render_window.Render()

        def _render(self) -> None:
            # Rendering a hidden native window can create an unwanted top-level
            # OpenGL window on some platforms. Qt will render it on the first
            # paint event; visible widgets are updated immediately for sliders.
            if not self._vtk_widget.isVisible():
                self._vtk_widget.update()
                return

            remaining = _FRAME_INTERVAL_SECONDS - (
                time.perf_counter() - self._last_render_started
            )
            if remaining <= 0.0 and not self._render_pending:
                self._perform_scheduled_render()
                return
            if self._render_pending:
                return
            self._render_pending = True
            delay_ms = max(1, int(math.ceil(max(remaining, 0.0) * 1000.0)))
            self._render_timer.start(delay_ms)


# Compatibility aliases keep the rest of the UI free to use the most natural
# naming convention without duplicating implementation.
VTKViewport = ViewportWidget
VTKViewportWidget = ViewportWidget
Viewport3D = ViewportWidget
