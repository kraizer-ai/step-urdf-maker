"""STEP assembly importer backed by Open Cascade (``cadquery-ocp``).

The importer deliberately returns a small, UI-independent data structure.  A
STEP/XCAF occurrence is represented by one :class:`StepPart`; its vertices are
already transformed into the assembly's zero-pose world frame and converted to
metres.  That convention lets the selection UI group CAD parts into URDF links
without having to understand Open Cascade locations.

XCAF is attempted first because it preserves product names, assembly
occurrences, and their placements.  Some STEP files do not contain a usable
XCAF product tree, so a plain ``STEPControl_Reader`` solid enumeration remains
as a conservative fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Sequence

import numpy as np


ProgressCallback = Callable[[int, int, str], None]


class StepLoadError(RuntimeError):
    """Raised when a STEP file cannot be read by either importer."""


def _stable_part_id(source_label: str, used: set[str]) -> str:
    """Create an occurrence-based ID that does not depend on mesh enumeration.

    XCAF label paths identify assembly occurrences.  Hashing the path keeps JSON
    project files compact while preventing an inserted/skipped earlier part from
    silently shifting every later ``part_0001`` style identifier.
    """

    digest = hashlib.sha1(source_label.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    base = f"part_{digest}"
    identifier = base
    suffix = 2
    while identifier in used:
        identifier = f"{base}_{suffix}"
        suffix += 1
    used.add(identifier)
    return identifier


@dataclass
class StepPart:
    """One selectable STEP occurrence in its assembled zero pose.

    ``vertices`` are an ``(N, 3)`` float array in metres.  ``triangles`` are an
    ``(M, 3)`` zero-based integer index array.  ``source_label`` is a stable,
    human-readable occurrence path (XCAF label entries where available).
    """

    id: str
    name: str
    vertices: np.ndarray
    triangles: np.ndarray
    color: tuple[float, float, float, float] | None = None
    source_label: str = ""
    feature_axes: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.name = str(self.name)
        self.source_label = str(self.source_label)
        self.feature_axes = [dict(item) for item in self.feature_axes]
        vertices = np.asarray(self.vertices, dtype=np.float64)
        if vertices.size == 0:
            vertices = np.empty((0, 3), dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        triangles = np.asarray(self.triangles, dtype=np.int64)
        if triangles.size == 0:
            triangles = np.empty((0, 3), dtype=np.int64)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (M, 3)")
        if triangles.size and (
            int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices)
        ):
            raise ValueError("triangle index is outside vertices")
        if self.color is not None:
            rgba = np.asarray(self.color, dtype=float).reshape(-1)
            if rgba.size == 3:
                rgba = np.append(rgba, 1.0)
            if rgba.size != 4:
                raise ValueError("color must contain RGB or RGBA values")
            self.color = tuple(float(value) for value in np.clip(rgba, 0.0, 1.0))
        self.vertices = vertices
        self.triangles = triangles

    @property
    def vertices_zero(self) -> np.ndarray:
        """Alias used by :class:`urdf_maker.model.ScenePart`."""

        return self.vertices

    @vertices_zero.setter
    def vertices_zero(self, value: np.ndarray) -> None:
        self.vertices = np.asarray(value, dtype=np.float64)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not len(self.vertices):
            return None
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def to_scene_part(self):
        """Convert to the application's core ``ScenePart`` lazily.

        The lazy import keeps this module useful as a standalone STEP reader and
        avoids an import cycle while the UI/model modules are initialized.
        """

        from .model import ScenePart

        kwargs: dict[str, Any] = {}
        if self.color is not None:
            kwargs["color"] = self.color
        return ScenePart(
            id=self.id,
            name=self.name,
            vertices_zero=self.vertices,
            triangles=self.triangles,
            feature_axes=self.feature_axes,
            **kwargs,
        )


@dataclass
class StepLoadResult:
    """Result and diagnostics from :func:`load_step`."""

    parts: list[StepPart] = field(default_factory=list)
    units: str = "m"
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.parts = list(self.parts)
        self.units = str(self.units)
        self.warnings = [str(item) for item in self.warnings]
        self.metadata = dict(self.metadata)

    def to_robot_project(self, name: str | None = None):
        """Create a parts-only ``RobotProject`` ready for link grouping."""

        from .model import RobotProject, sanitize_name

        source = self.metadata.get("source_path")
        display_name = name or (Path(source).stem if source else "step_robot")
        return RobotProject(
            name=sanitize_name(display_name, "robot"),
            parts=[part.to_scene_part() for part in self.parts],
            source_path=source,
            source_kind="step",
            metadata={**self.metadata, "step_warnings": list(self.warnings)},
        )


@dataclass
class _OcpApi:
    STEPCAFControl_Reader: Any
    STEPControl_Reader: Any
    IFSelect_RetDone: Any
    XCAFApp_Application: Any
    TDocStd_Document: Any
    TCollection_ExtendedString: Any
    TCollection_AsciiString: Any
    TColStd_SequenceOfAsciiString: Any
    XCAFDoc_DocumentTool: Any
    XCAFDoc_ShapeTool: Any
    XCAFDoc_ColorTool: Any
    XCAFDoc_ColorType: Any
    TDF_Label: Any
    TDF_LabelSequence: Any
    TDF_Tool: Any
    TDataStd_Name: Any
    TopLoc_Location: Any
    BRepMesh_IncrementalMesh: Any
    BRepAdaptor_Surface: Any
    BRep_Tool: Any
    TopExp_Explorer: Any
    TopoDS: Any
    TopAbs_FACE: Any
    TopAbs_SOLID: Any
    TopAbs_REVERSED: Any
    GeomAbs_Cylinder: Any
    Quantity_ColorRGBA: Any


def _import_ocp() -> _OcpApi:
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.GeomAbs import GeomAbs_Cylinder
        from OCP.Quantity import Quantity_ColorRGBA
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TCollection import (
            TCollection_AsciiString,
            TCollection_ExtendedString,
        )
        from OCP.TColStd import TColStd_SequenceOfAsciiString
        from OCP.TDF import TDF_Label, TDF_LabelSequence, TDF_Tool
        from OCP.TDataStd import TDataStd_Name
        from OCP.TDocStd import TDocStd_Document
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import (
            XCAFDoc_ColorTool,
            XCAFDoc_ColorType,
            XCAFDoc_DocumentTool,
            XCAFDoc_ShapeTool,
        )
    except (ImportError, OSError) as exc:  # OSError also covers missing DLLs.
        raise StepLoadError(
            "STEP support requires cadquery-ocp. Install the project dependencies "
            "and ensure the matching VTK runtime is available."
        ) from exc

    return _OcpApi(**locals())


def _label_name(api: _OcpApi, label: Any) -> str:
    if label is None or label.IsNull():
        return ""
    attribute = api.TDataStd_Name()
    if label.FindAttribute(api.TDataStd_Name.GetID_s(), attribute):
        return str(attribute.Get().ToExtString()).strip()
    return ""


def _label_entry(api: _OcpApi, label: Any) -> str:
    if label is None or label.IsNull():
        return ""
    entry = api.TCollection_AsciiString()
    api.TDF_Tool.Entry_s(label, entry)
    return str(entry.ToCString())


def _is_generated_occurrence_name(name: str) -> bool:
    upper = name.strip().upper()
    return upper.startswith("NAUO") and upper[4:].isdigit()


def _display_name(api: _OcpApi, occurrence: Any, definition: Any, index: int) -> str:
    occurrence_name = _label_name(api, occurrence)
    definition_name = _label_name(api, definition)
    if occurrence_name and not _is_generated_occurrence_name(occurrence_name):
        return occurrence_name
    return definition_name or occurrence_name or f"part_{index:04d}"


def _trsf_matrix(location: Any) -> np.ndarray:
    """Convert an OCCT location to a homogeneous numpy matrix."""

    transform = location.Transformation()
    matrix = np.eye(4, dtype=np.float64)
    for row in range(1, 4):
        for column in range(1, 5):
            matrix[row - 1, column - 1] = transform.Value(row, column)
    return matrix


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a 3-D affine transform without dispatching to BLAS.

    The matrices here are always 3x3. Calling NumPy's general matrix product
    needlessly initializes a BLAS runtime in the short-lived CAD worker, which
    is both slower for this size and vulnerable to unrelated Conda DLLs on an
    unactivated Windows ``PATH``.
    """

    result = np.empty((len(points), 3), dtype=np.float64)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    result[:, 0] = (
        x * matrix[0, 0]
        + y * matrix[0, 1]
        + z * matrix[0, 2]
        + matrix[0, 3]
    )
    result[:, 1] = (
        x * matrix[1, 0]
        + y * matrix[1, 1]
        + z * matrix[1, 2]
        + matrix[1, 3]
    )
    result[:, 2] = (
        x * matrix[2, 0]
        + y * matrix[2, 1]
        + z * matrix[2, 2]
        + matrix[2, 3]
    )
    return result


def _mesh_shape(
    api: _OcpApi,
    shape: Any,
    *,
    occurrence_location: Any,
    linear_deflection: float,
    angular_deflection: float,
    parallel: bool,
    scale_to_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a definition and apply face plus occurrence placements."""

    mesher = api.BRepMesh_IncrementalMesh(
        shape,
        float(linear_deflection),
        False,
        float(angular_deflection),
        bool(parallel),
    )
    if not mesher.IsDone():
        # ``Perform`` is harmless for bindings that defer construction work.
        mesher.Perform()

    occurrence_matrix = _trsf_matrix(occurrence_location)
    vertex_chunks: list[np.ndarray] = []
    triangle_chunks: list[np.ndarray] = []
    vertex_offset = 0
    explorer = api.TopExp_Explorer(shape, api.TopAbs_FACE)
    while explorer.More():
        face = api.TopoDS.Face_s(explorer.Current())
        face_location = api.TopLoc_Location()
        triangulation = api.BRep_Tool.Triangulation_s(face, face_location)
        if triangulation is not None and triangulation.NbNodes() > 0:
            nodes = np.empty((triangulation.NbNodes(), 3), dtype=np.float64)
            for node_index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(node_index)
                nodes[node_index - 1] = (point.X(), point.Y(), point.Z())

            nodes = _transform_points(nodes, _trsf_matrix(face_location))
            nodes = _transform_points(nodes, occurrence_matrix)
            nodes *= scale_to_m

            triangles = np.empty(
                (triangulation.NbTriangles(), 3), dtype=np.int64
            )
            reverse = face.Orientation() == api.TopAbs_REVERSED
            for triangle_index in range(1, triangulation.NbTriangles() + 1):
                n1, n2, n3 = triangulation.Triangle(triangle_index).Get()
                # Poly_Triangulation nodes are one-based.
                if reverse:
                    n2, n3 = n3, n2
                triangles[triangle_index - 1] = (
                    n1 - 1 + vertex_offset,
                    n2 - 1 + vertex_offset,
                    n3 - 1 + vertex_offset,
                )
            vertex_chunks.append(nodes)
            triangle_chunks.append(triangles)
            vertex_offset += len(nodes)
        explorer.Next()

    if not vertex_chunks:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
        )
    return np.vstack(vertex_chunks), np.vstack(triangle_chunks)


def _cylindrical_feature_axes(
    api: _OcpApi,
    shape: Any,
    *,
    occurrence_location: Any,
    scale_to_m: float,
) -> list[dict[str, Any]]:
    """Extract exact cylinder centerlines before STEP topology is discarded."""

    occurrence_matrix = _trsf_matrix(occurrence_location)
    result: list[dict[str, Any]] = []
    explorer = api.TopExp_Explorer(shape, api.TopAbs_FACE)
    while explorer.More():
        face = api.TopoDS.Face_s(explorer.Current())
        try:
            surface = api.BRepAdaptor_Surface(face, True)
            if surface.GetType() != api.GeomAbs_Cylinder:
                explorer.Next()
                continue
            cylinder = surface.Cylinder()
            axis = cylinder.Axis()
            location = axis.Location()
            direction = axis.Direction()
            origin_local = np.asarray(
                (location.X(), location.Y(), location.Z()),
                dtype=np.float64,
            )
            direction_local = np.asarray(
                (direction.X(), direction.Y(), direction.Z()),
                dtype=np.float64,
            )
            first_v = float(surface.FirstVParameter())
            last_v = float(surface.LastVParameter())
            if (
                math.isfinite(first_v)
                and math.isfinite(last_v)
                and abs(first_v) < 1.0e12
                and abs(last_v) < 1.0e12
            ):
                origin_local = (
                    origin_local
                    + direction_local * ((first_v + last_v) * 0.5)
                )
                axial_length = abs(last_v - first_v) * scale_to_m
            else:
                axial_length = 0.0
            origin_world = _transform_points(
                origin_local.reshape(1, 3),
                occurrence_matrix,
            )[0] * scale_to_m
            direction_world = occurrence_matrix[:3, :3] @ direction_local
            direction_world /= np.linalg.norm(direction_world)
            result.append(
                {
                    "kind": "cylinder",
                    "origin": origin_world.tolist(),
                    "direction": direction_world.tolist(),
                    "radius": float(cylinder.Radius()) * scale_to_m,
                    "length": float(axial_length),
                }
            )
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            pass
        explorer.Next()
    return result


def _label_color(api: _OcpApi, color_tool: Any, labels: Iterable[Any]) -> tuple | None:
    color_types = (
        api.XCAFDoc_ColorType.XCAFDoc_ColorSurf,
        api.XCAFDoc_ColorType.XCAFDoc_ColorGen,
        api.XCAFDoc_ColorType.XCAFDoc_ColorCurv,
    )
    for label in labels:
        if label is None or label.IsNull():
            continue
        for color_type in color_types:
            color = api.Quantity_ColorRGBA()
            if api.XCAFDoc_ColorTool.GetColor_s(label, color_type, color):
                rgb = color.GetRGB()
                return (rgb.Red(), rgb.Green(), rgb.Blue(), color.Alpha())
        shape = api.XCAFDoc_ShapeTool.GetShape_s(label)
        if shape.IsNull():
            continue
        for color_type in color_types:
            color = api.Quantity_ColorRGBA()
            if color_tool.GetInstanceColor(shape, color_type, color) or color_tool.GetColor(
                shape, color_type, color
            ):
                rgb = color.GetRGB()
                return (rgb.Red(), rgb.Green(), rgb.Blue(), color.Alpha())
    return None


def _file_units(api: _OcpApi, step_reader: Any) -> list[str]:
    lengths = api.TColStd_SequenceOfAsciiString()
    angles = api.TColStd_SequenceOfAsciiString()
    solid_angles = api.TColStd_SequenceOfAsciiString()
    try:
        step_reader.FileUnits(lengths, angles, solid_angles)
    except Exception:
        return []
    return [
        str(lengths.Value(index).ToCString())
        for index in range(1, lengths.Length() + 1)
    ]


def _system_scale_to_m(step_reader: Any) -> tuple[float, float]:
    """Return (OCCT system unit in millimetres, scale to metres)."""

    try:
        system_unit_mm = float(step_reader.SystemLengthUnit())
    except Exception:
        system_unit_mm = 1.0
    if not np.isfinite(system_unit_mm) or system_unit_mm <= 0.0:
        system_unit_mm = 1.0
    return system_unit_mm, system_unit_mm / 1000.0


def _load_xcaf(
    api: _OcpApi,
    path: Path,
    *,
    linear_deflection_m: float,
    angular_deflection: float,
    parallel: bool,
    progress: ProgressCallback | None,
) -> StepLoadResult:
    start = perf_counter()
    application = api.XCAFApp_Application.GetApplication_s()
    document = api.TDocStd_Document(api.TCollection_ExtendedString("urdf-maker"))
    application.NewDocument(api.TCollection_ExtendedString("MDTV-XCAF"), document)

    reader = api.STEPCAFControl_Reader()
    reader.SetColorMode(True)
    reader.SetNameMode(True)
    reader.SetLayerMode(True)
    reader.SetPropsMode(True)
    try:
        reader.SetMatMode(True)
    except AttributeError:  # Older OCCT releases do not expose materials mode.
        pass
    status = reader.ReadFile(str(path))
    if status != api.IFSelect_RetDone:
        raise StepLoadError(f"Open Cascade could not read STEP file (status: {status})")
    step_reader = reader.Reader()
    source_units = _file_units(api, step_reader)
    system_unit_mm, scale_to_m = _system_scale_to_m(step_reader)
    if not reader.Transfer(document):
        raise StepLoadError("Open Cascade read the STEP file but XCAF transfer failed")

    shape_tool = api.XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    color_tool = api.XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    roots = api.TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    if roots.IsEmpty():
        raise StepLoadError("STEP file contains no free XCAF shapes")

    leaves: list[tuple[Any, Any, Any, tuple[str, ...], tuple[str, ...]]] = []

    def collect(
        label: Any,
        parent_location: Any,
        label_path: tuple[str, ...],
        name_path: tuple[str, ...],
    ) -> None:
        referred = api.TDF_Label()
        is_reference = api.XCAFDoc_ShapeTool.GetReferredShape_s(label, referred)
        definition = referred if is_reference else label
        local_location = api.XCAFDoc_ShapeTool.GetLocation_s(label)
        world_location = parent_location.Multiplied(local_location)
        display_name = _display_name(api, label, definition, len(leaves) + 1)
        current_label_path = label_path + (_label_entry(api, label),)
        current_name_path = name_path + (display_name,)

        components = api.TDF_LabelSequence()
        if api.XCAFDoc_ShapeTool.GetComponents_s(definition, components, False):
            for component_index in range(1, components.Length() + 1):
                collect(
                    components.Value(component_index),
                    world_location,
                    current_label_path,
                    current_name_path,
                )
            return
        leaves.append(
            (label, definition, world_location, current_label_path, current_name_path)
        )

    identity = api.TopLoc_Location()
    for root_index in range(1, roots.Length() + 1):
        collect(roots.Value(root_index), identity, (), ())
    if not leaves:
        raise StepLoadError("XCAF assembly contains no leaf parts")

    warnings: list[str] = []
    parts: list[StepPart] = []
    used_part_ids: set[str] = set()
    occurrences: list[dict[str, Any]] = []
    # The mesher sees coordinates in OCCT's system unit (normally millimetres).
    linear_deflection_internal = linear_deflection_m / scale_to_m
    for index, (occurrence, definition, location, labels, names) in enumerate(
        leaves, start=1
    ):
        name = _display_name(api, occurrence, definition, index)
        source_label = "/".join(item for item in labels if item)
        if progress is not None:
            progress(index - 1, len(leaves), name)
        definition_shape = api.XCAFDoc_ShapeTool.GetShape_s(definition)
        if definition_shape.IsNull():
            warnings.append(f"Skipped {name!r}: XCAF definition has no shape")
            continue
        try:
            vertices, triangles = _mesh_shape(
                api,
                definition_shape,
                occurrence_location=location,
                linear_deflection=linear_deflection_internal,
                angular_deflection=angular_deflection,
                parallel=parallel,
                scale_to_m=scale_to_m,
            )
        except Exception as exc:
            warnings.append(f"Skipped {name!r}: triangulation failed ({exc})")
            continue
        if not len(triangles):
            warnings.append(f"Skipped {name!r}: shape produced no triangles")
            continue
        try:
            feature_axes = _cylindrical_feature_axes(
                api,
                definition_shape,
                occurrence_location=location,
                scale_to_m=scale_to_m,
            )
        except Exception:
            feature_axes = []
        identifier = _stable_part_id(source_label, used_part_ids)
        part = StepPart(
            id=identifier,
            name=name,
            vertices=vertices,
            triangles=triangles,
            color=_label_color(api, color_tool, (occurrence, definition)),
            source_label=source_label,
            feature_axes=feature_axes,
        )
        parts.append(part)
        occurrences.append(
            {
                "id": identifier,
                "name": name,
                "source_label": source_label,
                "definition_label": _label_entry(api, definition),
                "assembly_path": list(names),
            }
        )
    if progress is not None:
        progress(len(leaves), len(leaves), "STEP import complete")
    if not parts:
        raise StepLoadError("XCAF shapes were found but none could be triangulated")

    metadata: dict[str, Any] = {
        "source_path": str(path),
        "import_mode": "xcaf",
        "source_units": source_units,
        "system_unit_mm": system_unit_mm,
        "scale_to_m": scale_to_m,
        "root_count": roots.Length(),
        "occurrence_count": len(leaves),
        "part_count": len(parts),
        "linear_deflection_m": linear_deflection_m,
        "angular_deflection_rad": angular_deflection,
        "load_seconds": perf_counter() - start,
        "occurrences": occurrences,
    }
    all_vertices = [part.vertices for part in parts if len(part.vertices)]
    if all_vertices:
        metadata["bounds_m"] = [
            np.min([vertices.min(axis=0) for vertices in all_vertices], axis=0).tolist(),
            np.max([vertices.max(axis=0) for vertices in all_vertices], axis=0).tolist(),
        ]
    return StepLoadResult(parts=parts, units="m", warnings=warnings, metadata=metadata)


def _solid_shapes(api: _OcpApi, shapes: Sequence[Any]) -> list[tuple[int, int, Any]]:
    solids: list[tuple[int, int, Any]] = []
    for root_index, shape in enumerate(shapes, start=1):
        explorer = api.TopExp_Explorer(shape, api.TopAbs_SOLID)
        solid_index = 0
        while explorer.More():
            solid_index += 1
            solids.append((root_index, solid_index, api.TopoDS.Solid_s(explorer.Current())))
            explorer.Next()
    return solids


def _load_fallback(
    api: _OcpApi,
    path: Path,
    *,
    linear_deflection_m: float,
    angular_deflection: float,
    parallel: bool,
    progress: ProgressCallback | None,
    prior_warnings: Sequence[str] = (),
) -> StepLoadResult:
    start = perf_counter()
    reader = api.STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != api.IFSelect_RetDone:
        raise StepLoadError(f"Open Cascade could not read STEP file (status: {status})")
    source_units = _file_units(api, reader)
    system_unit_mm, scale_to_m = _system_scale_to_m(reader)
    transferred = reader.TransferRoots()
    if transferred <= 0:
        raise StepLoadError("STEP file has no transferable shape roots")
    root_shapes = [
        reader.Shape(index) for index in range(1, reader.NbShapes() + 1)
    ]
    root_shapes = [shape for shape in root_shapes if not shape.IsNull()]
    solids = _solid_shapes(api, root_shapes)
    if not solids:
        solids = [(index, 1, shape) for index, shape in enumerate(root_shapes, start=1)]

    parts: list[StepPart] = []
    used_part_ids: set[str] = set()
    warnings = list(prior_warnings)
    linear_deflection_internal = linear_deflection_m / scale_to_m
    for index, (root_index, solid_index, shape) in enumerate(solids, start=1):
        name = f"solid_{index:04d}"
        if progress is not None:
            progress(index - 1, len(solids), name)
        try:
            vertices, triangles = _mesh_shape(
                api,
                shape,
                occurrence_location=api.TopLoc_Location(),
                linear_deflection=linear_deflection_internal,
                angular_deflection=angular_deflection,
                parallel=parallel,
                scale_to_m=scale_to_m,
            )
        except Exception as exc:
            warnings.append(f"Skipped {name!r}: triangulation failed ({exc})")
            continue
        if not len(triangles):
            warnings.append(f"Skipped {name!r}: shape produced no triangles")
            continue
        source_label = f"root:{root_index}/solid:{solid_index}"
        try:
            feature_axes = _cylindrical_feature_axes(
                api,
                shape,
                occurrence_location=api.TopLoc_Location(),
                scale_to_m=scale_to_m,
            )
        except Exception:
            feature_axes = []
        parts.append(
            StepPart(
                id=_stable_part_id(source_label, used_part_ids),
                name=name,
                vertices=vertices,
                triangles=triangles,
                source_label=source_label,
                feature_axes=feature_axes,
            )
        )
    if progress is not None:
        progress(len(solids), len(solids), "STEP fallback import complete")
    if not parts:
        raise StepLoadError("STEP shapes were found but none could be triangulated")

    metadata: dict[str, Any] = {
        "source_path": str(path),
        "import_mode": "stepcontrol-fallback",
        "source_units": source_units,
        "system_unit_mm": system_unit_mm,
        "scale_to_m": scale_to_m,
        "root_count": len(root_shapes),
        "solid_count": len(solids),
        "part_count": len(parts),
        "linear_deflection_m": linear_deflection_m,
        "angular_deflection_rad": angular_deflection,
        "load_seconds": perf_counter() - start,
    }
    all_vertices = [part.vertices for part in parts if len(part.vertices)]
    if all_vertices:
        metadata["bounds_m"] = [
            np.min([vertices.min(axis=0) for vertices in all_vertices], axis=0).tolist(),
            np.max([vertices.max(axis=0) for vertices in all_vertices], axis=0).tolist(),
        ]
    return StepLoadResult(parts=parts, units="m", warnings=warnings, metadata=metadata)


def load_step(
    path: str | Path,
    *,
    linear_deflection: float = 5e-4,
    angular_deflection: float = 0.35,
    parallel: bool = True,
    prefer_xcaf: bool = True,
    progress: ProgressCallback | None = None,
) -> StepLoadResult:
    """Load *path* into selectable, assembled triangle meshes.

    Parameters use SI units: ``linear_deflection`` is in metres and
    ``angular_deflection`` is in radians.  A coarser linear value (for example
    0.001) makes large mechanical assemblies faster to preview.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.casefold() not in {".step", ".stp"}:
        raise StepLoadError(f"Not a STEP file: {source.name}")
    linear_deflection = float(linear_deflection)
    angular_deflection = float(angular_deflection)
    if not np.isfinite(linear_deflection) or linear_deflection <= 0.0:
        raise ValueError("linear_deflection must be a positive finite value")
    if not np.isfinite(angular_deflection) or angular_deflection <= 0.0:
        raise ValueError("angular_deflection must be a positive finite value")

    api = _import_ocp()
    xcaf_warning: list[str] = []
    if prefer_xcaf:
        try:
            return _load_xcaf(
                api,
                source,
                linear_deflection_m=linear_deflection,
                angular_deflection=angular_deflection,
                parallel=parallel,
                progress=progress,
            )
        except Exception as exc:
            xcaf_warning.append(f"XCAF assembly import failed; used solid fallback: {exc}")
    try:
        return _load_fallback(
            api,
            source,
            linear_deflection_m=linear_deflection,
            angular_deflection=angular_deflection,
            parallel=parallel,
            progress=progress,
            prior_warnings=xcaf_warning,
        )
    except Exception as fallback_error:
        if xcaf_warning:
            raise StepLoadError(
                f"STEP import failed in both XCAF and fallback modes. "
                f"{xcaf_warning[0]} Fallback error: {fallback_error}"
            ) from fallback_error
        if isinstance(fallback_error, StepLoadError):
            raise
        raise StepLoadError(f"STEP fallback import failed: {fallback_error}") from fallback_error


def load_step_project(path: str | Path, **kwargs: Any):
    """Convenience wrapper returning a parts-only ``RobotProject``."""

    return load_step(path, **kwargs).to_robot_project()


__all__ = [
    "ProgressCallback",
    "StepLoadError",
    "StepLoadResult",
    "StepPart",
    "load_step",
    "load_step_project",
]
