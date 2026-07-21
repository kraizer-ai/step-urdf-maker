"""URDF import/export and dependency-free triangle mesh readers.

Only NumPy and the Python standard library are required.  The mesh readers are
deliberately small but cover the formats most commonly referenced by URDF:
binary/ASCII STL, OBJ, PLY, and geometry-oriented COLLADA (DAE).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
import os
from pathlib import Path
import struct
from typing import Any
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np

from .model import (
    JointSpec,
    LinkSpec,
    ProjectValidationError,
    RobotProject,
    ScenePart,
    apply_transform,
    sanitize_name,
    transform_matrix,
)


class MeshFormatError(ValueError):
    """Raised when a referenced mesh is malformed or unsupported."""


def _floats(text: str | None, count: int | None = None, default: Iterable[float] = ()) -> np.ndarray:
    values = np.fromstring(text or "", sep=" ", dtype=float)
    if not len(values):
        values = np.asarray(tuple(default), dtype=float)
    if count is not None and len(values) != count:
        raise ValueError(f"Expected {count} numeric values; got {len(values)}")
    return values


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element if _local_name(child.tag) == name]


def _first(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _unique(value: str, used: set[str], prefix: str) -> str:
    base = sanitize_name(value, prefix)
    result = base
    number = 2
    while result in used:
        result = f"{base}_{number}"
        number += 1
    used.add(result)
    return result


def _package_name(package_xml: Path) -> str | None:
    try:
        root = ET.parse(package_xml).getroot()
        node = root.find("name")
        return node.text.strip() if node is not None and node.text else None
    except (OSError, ET.ParseError):
        return None


def resolve_mesh_path(
    filename: str,
    urdf_path: str | Path,
    package_dirs: Mapping[str, str | Path] | Iterable[str | Path] | None = None,
) -> Path:
    """Resolve relative, ``file://``, and ``package://`` URDF mesh paths.

    ``package_dirs`` accepts either ``{package_name: package_root}`` or a list
    containing package roots / directories containing package roots.
    """

    urdf_file = Path(urdf_path).expanduser().resolve()
    value = unquote(str(filename).strip())
    if value.startswith("file://"):
        parsed = urlparse(value)
        raw = unquote(parsed.path)
        if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            raw = f"//{parsed.netloc}{raw}"
        return Path(raw).expanduser().resolve()
    if not value.startswith("package://"):
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = urdf_file.parent / path
        return path.resolve()

    remainder = value[len("package://") :]
    if "/" not in remainder:
        raise FileNotFoundError(f"Invalid package URI: {filename}")
    package, relative = remainder.split("/", 1)
    candidates: list[Path] = []
    if isinstance(package_dirs, Mapping):
        configured = package_dirs.get(package)
        if configured is not None:
            candidates.append(Path(configured) / relative)
        roots: list[Path] = []
    else:
        if isinstance(package_dirs, (str, os.PathLike)):
            roots = [Path(package_dirs).expanduser()]
        else:
            roots = [Path(item).expanduser() for item in (package_dirs or ())]
    for root in roots:
        candidates.append(root / package / relative)
        if root.name == package:
            candidates.append(root / relative)

    # The URDF commonly lives in <package>/urdf, so inspect all ancestors.
    for ancestor in (urdf_file.parent, *urdf_file.parents):
        package_xml = ancestor / "package.xml"
        if package_xml.is_file() and _package_name(package_xml) == package:
            candidates.append(ancestor / relative)
        candidates.append(ancestor / package / relative)

    for item in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep):
        if item:
            root = Path(item).expanduser()
            candidates.extend((root / package / relative, root / relative))
    for item in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if item:
            candidates.append(Path(item).expanduser() / "share" / package / relative)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    locations = "\n  ".join(str(item) for item in candidates)
    raise FileNotFoundError(f"Could not resolve {filename!r}. Tried:\n  {locations}")


def _triangle_soup(triangles: list[list[list[float]]]) -> tuple[np.ndarray, np.ndarray]:
    if not triangles:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
    vertices = np.asarray(triangles, dtype=float).reshape(-1, 3)
    indices = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    return vertices, indices


def _load_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        expected = 84 + count * 50
        if expected <= len(data) and (expected == len(data) or not data[:5].lower() == b"solid"):
            vertices = np.empty((count * 3, 3), dtype=float)
            for index in range(count):
                values = struct.unpack_from("<12f", data, 84 + index * 50)
                vertices[index * 3 : index * 3 + 3] = np.asarray(values[3:12]).reshape(3, 3)
            return vertices, np.arange(count * 3, dtype=np.int64).reshape(-1, 3)

    triangles: list[list[list[float]]] = []
    current: list[list[float]] = []
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - bytes.decode is intentionally forgiving
        raise MeshFormatError(f"Cannot decode STL {path}: {exc}") from exc
    for line in text.splitlines():
        words = line.strip().split()
        if len(words) >= 4 and words[0].lower() == "vertex":
            current.append([float(words[1]), float(words[2]), float(words[3])])
            if len(current) == 3:
                triangles.append(current)
                current = []
    if not triangles:
        raise MeshFormatError(f"No triangles found in STL: {path}")
    return _triangle_soup(triangles)


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            words = line.strip().split()
            if not words or words[0].startswith("#"):
                continue
            if words[0] == "v" and len(words) >= 4:
                vertices.append([float(words[1]), float(words[2]), float(words[3])])
            elif words[0] == "f" and len(words) >= 4:
                face: list[int] = []
                for token in words[1:]:
                    raw = int(token.split("/", 1)[0])
                    face.append(raw - 1 if raw > 0 else len(vertices) + raw)
                for index in range(1, len(face) - 1):
                    triangles.append([face[0], face[index], face[index + 1]])
    vertex_array = np.asarray(vertices, dtype=float).reshape(-1, 3)
    triangle_array = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if triangle_array.size and (triangle_array.min() < 0 or triangle_array.max() >= len(vertex_array)):
        raise MeshFormatError(f"OBJ face references an invalid vertex: {path}")
    return vertex_array, triangle_array


_PLY_SCALARS = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def _load_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    marker = b"end_header"
    marker_at = data.find(marker)
    if marker_at < 0:
        raise MeshFormatError(f"PLY has no end_header: {path}")
    body_at = data.find(b"\n", marker_at)
    if body_at < 0:
        body_at = len(data)
    else:
        body_at += 1
    header = data[:body_at].decode("ascii", errors="strict").splitlines()
    if not header or header[0].strip() != "ply":
        raise MeshFormatError(f"Not a PLY file: {path}")
    format_name = ""
    elements: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in header[1:]:
        words = line.strip().split()
        if not words:
            continue
        if words[0] == "format":
            format_name = words[1]
        elif words[0] == "element":
            current = {"name": words[1], "count": int(words[2]), "properties": []}
            elements.append(current)
        elif words[0] == "property" and current is not None:
            if words[1] == "list":
                current["properties"].append((words[-1], "list", words[2], words[3]))
            else:
                current["properties"].append((words[-1], "scalar", words[1]))

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    if format_name == "ascii":
        rows = iter(data[body_at:].decode("ascii", errors="replace").splitlines())
        for element in elements:
            for _ in range(element["count"]):
                values = next(rows).split()
                cursor = 0
                record: dict[str, Any] = {}
                for prop in element["properties"]:
                    if prop[1] == "scalar":
                        record[prop[0]] = float(values[cursor])
                        cursor += 1
                    else:
                        count = int(values[cursor])
                        cursor += 1
                        record[prop[0]] = [int(item) for item in values[cursor : cursor + count]]
                        cursor += count
                if element["name"] == "vertex":
                    vertices.append([record["x"], record["y"], record["z"]])
                elif element["name"] == "face":
                    face = record.get("vertex_indices", record.get("vertex_index", []))
                    faces.append(face)
    elif format_name in {"binary_little_endian", "binary_big_endian"}:
        endian = "<" if format_name == "binary_little_endian" else ">"
        cursor = body_at

        def scalar(kind: str) -> float | int:
            nonlocal cursor
            code = _PLY_SCALARS.get(kind)
            if code is None:
                raise MeshFormatError(f"Unsupported PLY scalar type {kind!r}")
            value = struct.unpack_from(endian + code, data, cursor)[0]
            cursor += struct.calcsize(code)
            return value

        for element in elements:
            for _ in range(element["count"]):
                record = {}
                for prop in element["properties"]:
                    if prop[1] == "scalar":
                        record[prop[0]] = scalar(prop[2])
                    else:
                        count = int(scalar(prop[2]))
                        record[prop[0]] = [int(scalar(prop[3])) for _ in range(count)]
                if element["name"] == "vertex":
                    vertices.append([record["x"], record["y"], record["z"]])
                elif element["name"] == "face":
                    faces.append(record.get("vertex_indices", record.get("vertex_index", [])))
    else:
        raise MeshFormatError(f"Unsupported PLY format {format_name!r}: {path}")

    triangles: list[list[int]] = []
    for face in faces:
        for index in range(1, len(face) - 1):
            triangles.append([face[0], face[index], face[index + 1]])
    return (
        np.asarray(vertices, dtype=float).reshape(-1, 3),
        np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
    )


def _dae_geometry(mesh: ET.Element) -> tuple[np.ndarray, np.ndarray]:
    sources: dict[str, np.ndarray] = {}
    for source in _children(mesh, "source"):
        identifier = source.get("id", "")
        array_node = next(
            (item for item in source.iter() if _local_name(item.tag) == "float_array"), None
        )
        if array_node is None:
            continue
        raw = _floats(array_node.text)
        accessor = next(
            (item for item in source.iter() if _local_name(item.tag) == "accessor"), None
        )
        stride = int(accessor.get("stride", "1")) if accessor is not None else 1
        offset = int(accessor.get("offset", "0")) if accessor is not None else 0
        count = int(accessor.get("count", str((len(raw) - offset) // stride))) if accessor is not None else len(raw) // stride
        sources[identifier] = raw[offset : offset + count * stride].reshape(count, stride)

    vertices_sources: dict[str, str] = {}
    for vertices in _children(mesh, "vertices"):
        for input_node in _children(vertices, "input"):
            if input_node.get("semantic") == "POSITION":
                vertices_sources[vertices.get("id", "")] = input_node.get("source", "").lstrip("#")

    output_vertices: list[np.ndarray] = []
    output_triangles: list[list[int]] = []
    vertex_cache: dict[tuple[str, int], int] = {}

    def output_index(source_id: str, source_index: int) -> int:
        key = (source_id, source_index)
        if key not in vertex_cache:
            values = sources[source_id][source_index]
            if len(values) < 3:
                raise MeshFormatError("COLLADA position source has fewer than three values")
            vertex_cache[key] = len(output_vertices)
            output_vertices.append(values[:3].astype(float))
        return vertex_cache[key]

    for primitive in mesh:
        kind = _local_name(primitive.tag)
        if kind not in {"triangles", "polylist", "polygons"}:
            continue
        inputs = _children(primitive, "input")
        if not inputs:
            continue
        stride = max(int(item.get("offset", "0")) for item in inputs) + 1
        position_input: tuple[int, str] | None = None
        for item in inputs:
            semantic = item.get("semantic")
            source_id = item.get("source", "").lstrip("#")
            if semantic == "VERTEX":
                source_id = vertices_sources.get(source_id, source_id)
                position_input = (int(item.get("offset", "0")), source_id)
            elif semantic == "POSITION":
                position_input = (int(item.get("offset", "0")), source_id)
        if position_input is None:
            continue
        position_offset, source_id = position_input

        if kind == "triangles":
            p_node = _first(primitive, "p")
            packed = np.fromstring(p_node.text or "", sep=" ", dtype=np.int64) if p_node is not None else np.empty(0, dtype=np.int64)
            indices = packed.reshape(-1, stride)[:, position_offset]
            for start in range(0, len(indices) - 2, 3):
                output_triangles.append(
                    [output_index(source_id, int(item)) for item in indices[start : start + 3]]
                )
        elif kind == "polylist":
            p_node, counts_node = _first(primitive, "p"), _first(primitive, "vcount")
            if p_node is None or counts_node is None:
                continue
            packed = np.fromstring(p_node.text or "", sep=" ", dtype=np.int64).reshape(-1, stride)
            indices = packed[:, position_offset]
            counts = np.fromstring(counts_node.text or "", sep=" ", dtype=np.int64)
            cursor = 0
            for count in counts:
                face = indices[cursor : cursor + int(count)]
                cursor += int(count)
                for index in range(1, len(face) - 1):
                    output_triangles.append(
                        [
                            output_index(source_id, int(face[0])),
                            output_index(source_id, int(face[index])),
                            output_index(source_id, int(face[index + 1])),
                        ]
                    )
        else:  # polygons
            for p_node in _children(primitive, "p"):
                packed = np.fromstring(p_node.text or "", sep=" ", dtype=np.int64).reshape(-1, stride)
                face = packed[:, position_offset]
                for index in range(1, len(face) - 1):
                    output_triangles.append(
                        [
                            output_index(source_id, int(face[0])),
                            output_index(source_id, int(face[index])),
                            output_index(source_id, int(face[index + 1])),
                        ]
                    )
    return (
        np.asarray(output_vertices, dtype=float).reshape(-1, 3),
        np.asarray(output_triangles, dtype=np.int64).reshape(-1, 3),
    )


def _dae_node_transform(node: ET.Element) -> np.ndarray:
    result = np.eye(4, dtype=float)
    for item in node:
        kind = _local_name(item.tag)
        values = _floats(item.text)
        transform = np.eye(4, dtype=float)
        if kind == "matrix" and len(values) == 16:
            transform = values.reshape(4, 4).T
        elif kind == "translate" and len(values) >= 3:
            transform[:3, 3] = values[:3]
        elif kind == "scale" and len(values) >= 3:
            transform[:3, :3] = np.diag(values[:3])
        elif kind == "rotate" and len(values) >= 4:
            axis = values[:3]
            norm = np.linalg.norm(axis)
            if norm:
                x, y, z = axis / norm
                angle = math.radians(values[3])
                c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
                transform[:3, :3] = np.array(
                    (
                        (t * x * x + c, t * x * y - s * z, t * x * z + s * y),
                        (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
                        (t * x * z - s * y, t * y * z + s * x, t * z * z + c),
                    )
                )
        else:
            continue
        result = result @ transform
    return result


def _load_dae(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    unit_scale = 1.0
    up_axis = "Z_UP"
    asset = next((item for item in root if _local_name(item.tag) == "asset"), None)
    if asset is not None:
        unit = _first(asset, "unit")
        if unit is not None:
            unit_scale = float(unit.get("meter", "1"))
        up = _first(asset, "up_axis")
        if up is not None and up.text:
            up_axis = up.text.strip()

    geometries: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for geometry in (item for item in root.iter() if _local_name(item.tag) == "geometry"):
        mesh = _first(geometry, "mesh")
        if mesh is not None:
            geometries[geometry.get("id", "")] = _dae_geometry(mesh)

    instances: list[tuple[str, np.ndarray]] = []

    def visit(node: ET.Element, parent: np.ndarray) -> None:
        current = parent @ _dae_node_transform(node)
        for item in node:
            kind = _local_name(item.tag)
            if kind == "instance_geometry":
                instances.append((item.get("url", "").lstrip("#"), current))
            elif kind == "node":
                visit(item, current)

    visual_scenes = [item for item in root.iter() if _local_name(item.tag) == "visual_scene"]
    for scene in visual_scenes[:1]:
        for node in _children(scene, "node"):
            visit(node, np.eye(4))
    if not instances:
        instances = [(identifier, np.eye(4)) for identifier in geometries]

    vertex_arrays: list[np.ndarray] = []
    triangle_arrays: list[np.ndarray] = []
    offset = 0
    for identifier, matrix in instances:
        if identifier not in geometries:
            continue
        vertices, triangles = geometries[identifier]
        transformed = apply_transform(vertices, matrix)
        vertex_arrays.append(transformed)
        triangle_arrays.append(triangles + offset)
        offset += len(vertices)
    if not vertex_arrays:
        raise MeshFormatError(f"No triangle geometry found in COLLADA: {path}")
    vertices = np.vstack(vertex_arrays) * unit_scale
    triangles = np.vstack(triangle_arrays)
    if up_axis == "Y_UP":
        vertices = vertices[:, (0, 2, 1)] * np.array((1.0, -1.0, 1.0))
    elif up_axis == "X_UP":
        vertices = vertices[:, (2, 1, 0)] * np.array((-1.0, 1.0, 1.0))
    return vertices, triangles


def load_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a triangle mesh without changing its native coordinate units."""

    mesh_path = Path(path)
    suffix = mesh_path.suffix.lower()
    if suffix == ".stl":
        return _load_stl(mesh_path)
    if suffix == ".obj":
        return _load_obj(mesh_path)
    if suffix == ".ply":
        return _load_ply(mesh_path)
    if suffix in {".dae", ".collada"}:
        return _load_dae(mesh_path)
    raise MeshFormatError(f"Unsupported mesh format {suffix!r}: {mesh_path}")


def _box(size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = size / 2.0
    vertices = np.array(
        [
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices, triangles


def _cylinder(radius: float, length: float, segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    rings = [
        np.c_[radius * np.cos(angles), radius * np.sin(angles), np.full(segments, z)]
        for z in (-length / 2.0, length / 2.0)
    ]
    vertices = np.vstack((*rings, [[0.0, 0.0, -length / 2.0], [0.0, 0.0, length / 2.0]]))
    bottom, top = segments * 2, segments * 2 + 1
    triangles: list[list[int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.extend(
            ([index, nxt, segments + nxt], [index, segments + nxt, segments + index])
        )
        triangles.extend(([bottom, nxt, index], [top, segments + index, segments + nxt]))
    return vertices, np.asarray(triangles, dtype=np.int64)


def _sphere(radius: float, rings: int = 16, segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
    vertices = [[0.0, 0.0, radius]]
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append(
                [radius * math.sin(phi) * math.cos(theta), radius * math.sin(phi) * math.sin(theta), radius * math.cos(phi)]
            )
    vertices.append([0.0, 0.0, -radius])
    bottom = len(vertices) - 1
    triangles: list[list[int]] = []
    for segment in range(segments):
        nxt = (segment + 1) % segments
        triangles.append([0, 1 + segment, 1 + nxt])
        for ring in range(rings - 2):
            first = 1 + ring * segments
            second = first + segments
            triangles.extend(
                ([first + segment, second + segment, second + nxt], [first + segment, second + nxt, first + nxt])
            )
        last = 1 + (rings - 2) * segments
        triangles.append([last + segment, bottom, last + nxt])
    return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64)


def _origin(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if element is None:
        return np.zeros(3), np.zeros(3)
    return (
        _floats(element.get("xyz"), 3, (0.0, 0.0, 0.0)),
        _floats(element.get("rpy"), 3, (0.0, 0.0, 0.0)),
    )


def _geometry(
    geometry: ET.Element,
    urdf_path: Path,
    package_dirs: Mapping[str, str | Path] | Iterable[str | Path] | None,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    mesh = geometry.find("mesh")
    if mesh is not None:
        filename = mesh.get("filename") or mesh.get("url")
        if not filename:
            raise MeshFormatError("URDF mesh has no filename")
        path = resolve_mesh_path(filename, urdf_path, package_dirs)
        vertices, triangles = load_mesh(path)
        scale_values = _floats(mesh.get("scale"), default=(1.0, 1.0, 1.0))
        if len(scale_values) == 1:
            scale_values = np.repeat(scale_values, 3)
        if len(scale_values) != 3:
            raise MeshFormatError(f"Mesh scale must have one or three values: {filename}")
        return vertices * scale_values, triangles, str(path)
    box = geometry.find("box")
    if box is not None:
        vertices, triangles = _box(_floats(box.get("size"), 3))
        return vertices, triangles, None
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        vertices, triangles = _cylinder(float(cylinder.get("radius", "0")), float(cylinder.get("length", "0")))
        return vertices, triangles, None
    sphere = geometry.find("sphere")
    if sphere is not None:
        vertices, triangles = _sphere(float(sphere.get("radius", "0")))
        return vertices, triangles, None
    raise MeshFormatError("Unsupported or empty URDF geometry")


def load_urdf(
    path: str | Path,
    package_dirs: Mapping[str, str | Path] | Iterable[str | Path] | None = None,
    *,
    strict: bool = False,
) -> RobotProject:
    """Load an arbitrary URDF into an editable :class:`RobotProject`.

    Missing/unsupported visuals are recorded in ``project.metadata['warnings']``
    and skipped unless ``strict=True``.  If a link has no visual, its collision
    geometry is loaded as a display fallback.
    """

    urdf_path = Path(path).expanduser().resolve()
    robot_node = ET.parse(urdf_path).getroot()
    if _local_name(robot_node.tag) != "robot":
        raise ValueError(f"Root element is not <robot>: {urdf_path}")
    robot_name = robot_node.get("name") or urdf_path.stem
    links: dict[str, LinkSpec] = {}
    for link_node in robot_node.findall("link"):
        name = link_node.get("name")
        if name:
            links[name] = LinkSpec(name)

    joints: list[JointSpec] = []
    for index, node in enumerate(robot_node.findall("joint")):
        parent_node, child_node = node.find("parent"), node.find("child")
        if parent_node is None or child_node is None:
            if strict:
                raise ValueError("URDF joint is missing parent or child")
            continue
        origin_xyz, origin_rpy = _origin(node.find("origin"))
        axis_node = node.find("axis")
        axis = _floats(axis_node.get("xyz") if axis_node is not None else None, 3, (1.0, 0.0, 0.0))
        limit = node.find("limit")
        joint_type = node.get("type", "fixed")
        kwargs: dict[str, Any] = {}
        if limit is not None:
            kwargs = {
                "lower": float(limit.get("lower")) if limit.get("lower") is not None else None,
                "upper": float(limit.get("upper")) if limit.get("upper") is not None else None,
                "effort": float(limit.get("effort", "0")),
                "velocity": float(limit.get("velocity", "0")),
            }
        joints.append(
            JointSpec(
                name=node.get("name") or f"joint_{index + 1}",
                type=joint_type,
                parent=parent_node.get("link", ""),
                child=child_node.get("link", ""),
                origin_xyz=origin_xyz,
                origin_rpy=origin_rpy,
                axis=axis,
                **kwargs,
            )
        )

    child_links = {joint.child for joint in joints}
    roots = [name for name in links if name not in child_links]
    root_link = roots[0] if roots else (next(iter(links), None))
    project = RobotProject(
        name=robot_name,
        links=links,
        joints=joints,
        root_link=root_link,
        source_path=str(urdf_path),
        source_kind="urdf",
        metadata={"warnings": [], "mesh_files": []},
    )
    warnings: list[str] = project.metadata["warnings"]
    if any(link.find("inertial") is not None for link in robot_node.findall("link")):
        warnings.append(
            "원본 URDF의 inertial 값은 보기에는 영향을 주지 않으며 다시 내보낼 때 보존되지 않습니다."
        )
    if any(link.find("collision") is not None for link in robot_node.findall("link")):
        warnings.append(
            "원본 URDF의 collision 형상은 별도로 보존되지 않습니다. 다시 내보내면 visual 메시가 collision으로 사용됩니다."
        )
    joint_extension_tags = {"mimic", "dynamics", "safety_controller", "calibration"}
    has_joint_extensions = any(
        any(joint.find(tag) is not None for tag in joint_extension_tags)
        for joint in robot_node.findall("joint")
    )
    has_robot_extensions = any(
        robot_node.find(tag) is not None
        for tag in ("transmission", "gazebo", "ros2_control")
    )
    if has_joint_extensions or has_robot_extensions:
        warnings.append(
            "mimic/dynamics/transmission/Gazebo/ros2_control 등의 확장 태그는 편집 모델에 보존되지 않습니다."
        )
    if len(roots) != 1 and links:
        warnings.append(f"URDF has {len(roots)} root candidates; using {root_link!r}")
    try:
        zero_fk = project.forward_kinematics(zero=True)
    except ProjectValidationError as exc:
        if strict:
            raise
        warnings.extend(exc.errors)
        # Keep malformed external URDFs inspectable: known connected links use
        # their normal transforms, while unreachable roots remain at identity.
        zero_fk = {name: np.eye(4) for name in links}
        changed = True
        # At most |links| relaxation passes are useful for an acyclic graph.
        # The cap also keeps a malformed transform-bearing cycle inspectable.
        passes = 0
        while changed and passes < max(1, len(links)):
            changed = False
            passes += 1
            for joint in joints:
                if joint.parent in zero_fk and joint.child in links:
                    candidate = zero_fk[joint.parent] @ joint.transform(0.0)
                    if not np.array_equal(zero_fk[joint.child], candidate):
                        zero_fk[joint.child] = candidate
                        changed = True

    global_materials: dict[str, np.ndarray] = {}
    for material in robot_node.findall("material"):
        color_node = material.find("color")
        if material.get("name") and color_node is not None:
            global_materials[material.get("name", "")] = _floats(
                color_node.get("rgba"), 4, (0.72, 0.74, 0.78, 1.0)
            )

    used_part_ids: set[str] = set()
    for link_node in robot_node.findall("link"):
        link_name = link_node.get("name")
        if not link_name or link_name not in links:
            continue
        visuals = link_node.findall("visual")
        source_nodes = visuals if visuals else link_node.findall("collision")
        for visual_index, visual in enumerate(source_nodes):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            display_name = visual.get("name") or f"{link_name}_visual_{visual_index + 1}"
            color = np.array((0.72, 0.74, 0.78, 1.0), dtype=float)
            material = visual.find("material")
            if material is not None:
                color_node = material.find("color")
                if color_node is not None:
                    color = _floats(color_node.get("rgba"), 4, color)
                elif material.get("name") in global_materials:
                    color = global_materials[material.get("name", "")].copy()
            try:
                vertices, triangles, mesh_file = _geometry(geometry, urdf_path, package_dirs)
                if mesh_file:
                    project.metadata["mesh_files"].append(mesh_file)
                visual_xyz, visual_rpy = _origin(visual.find("origin"))
                local_to_world = zero_fk.get(link_name, np.eye(4)) @ transform_matrix(
                    visual_xyz, visual_rpy
                )
                vertices_zero = apply_transform(vertices, local_to_world)
                identifier = _unique(display_name, used_part_ids, "part")
                part = ScenePart(
                    identifier,
                    display_name,
                    vertices_zero,
                    triangles,
                    color,
                    link_name,
                )
                project.parts[identifier] = part
                project.links[link_name].part_ids.append(identifier)
            except (OSError, ValueError, ET.ParseError) as exc:
                message = f"{link_name}/{display_name}: {exc}"
                if strict:
                    raise MeshFormatError(message) from exc
                warnings.append(message)
    return project


def _format(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _write_binary_stl(path: Path, vertices: np.ndarray, triangles: np.ndarray) -> None:
    header = b"STEP URDF Maker binary STL"[:80].ljust(80, b"\0")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(struct.pack("<I", len(triangles)))
        for triangle in triangles:
            points = vertices[np.asarray(triangle, dtype=np.int64)]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            norm = float(np.linalg.norm(normal))
            if norm:
                normal /= norm
            else:
                normal[:] = 0.0
            stream.write(struct.pack("<12fH", *normal, *points.reshape(-1), 0))


def _name_map(names: Iterable[str], prefix: str) -> dict[str, str]:
    used: set[str] = set()
    return {name: _unique(name, used, prefix) for name in names}


def _add_inertial(link_node: ET.Element, vertices: np.ndarray, density: float) -> None:
    minimum, maximum = vertices.min(axis=0), vertices.max(axis=0)
    size = maximum - minimum
    effective_size = np.maximum(size, 1e-6)
    center = (minimum + maximum) / 2.0
    mass = max(float(density) * float(np.prod(effective_size)), 1e-9)
    x, y, z = effective_size
    ixx = mass * (y * y + z * z) / 12.0
    iyy = mass * (x * x + z * z) / 12.0
    izz = mass * (x * x + y * y) / 12.0
    inertial = ET.SubElement(link_node, "inertial")
    ET.SubElement(inertial, "origin", xyz=_format(center), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.12g}")
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.12g}",
        ixy="0",
        ixz="0",
        iyy=f"{iyy:.12g}",
        iyz="0",
        izz=f"{izz:.12g}",
    )


def export_urdf(
    project: RobotProject,
    output_dir: str | Path,
    *,
    package_name: str | None = None,
    robot_name: str | None = None,
    urdf_filename: str | None = None,
    include_collision: bool = True,
    include_inertial: bool = False,
    add_inertial: bool | None = None,
    density: float = 500.0,
) -> Path:
    """Export a self-contained ROS package with link-local binary STL meshes.

    The returned path points to ``<output_dir>/urdf/<name>.urdf``.  Invalid
    display names are sanitized consistently without mutating the project.
    """

    if add_inertial is not None:  # friendly alias used by early UI prototypes
        include_inertial = bool(add_inertial)
    project.assert_valid(check_names=False)
    package_dir = Path(output_dir).expanduser().resolve()
    meshes_dir = package_dir / "meshes"
    urdf_dir = package_dir / "urdf"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    urdf_dir.mkdir(parents=True, exist_ok=True)

    safe_robot = sanitize_name(robot_name or project.name, "robot")
    raw_package = package_name or f"{safe_robot.lower()}_description"
    safe_package = sanitize_name(raw_package.lower(), "robot_description").replace(".", "_").replace("-", "_")
    if not safe_package[0].isalpha():
        safe_package = "robot_" + safe_package
    link_names = _name_map(project.links.keys(), "link")
    joint_names = _name_map((joint.name for joint in project.joints), "joint")

    robot_node = ET.Element("robot", name=safe_robot)
    mesh_records: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for original_name, link in project.links.items():
        exported_name = link_names[original_name]
        link_node = ET.SubElement(robot_node, "link", name=exported_name)
        vertices, triangles = project.link_vertices_local(original_name)
        if len(triangles):
            mesh_file = f"{exported_name}.stl"
            _write_binary_stl(meshes_dir / mesh_file, vertices, triangles)
            uri = f"package://{safe_package}/meshes/{mesh_file}"
            visual = ET.SubElement(link_node, "visual", name=f"{exported_name}_visual")
            ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
            geometry = ET.SubElement(visual, "geometry")
            ET.SubElement(geometry, "mesh", filename=uri, scale="1 1 1")
            colors = [project.parts[item].color for item in link.part_ids if item in project.parts]
            rgba = np.mean(colors, axis=0) if colors else np.array((0.72, 0.74, 0.78, 1.0))
            material = ET.SubElement(visual, "material", name=f"{exported_name}_material")
            ET.SubElement(material, "color", rgba=_format(rgba))
            if include_collision:
                collision = ET.SubElement(link_node, "collision", name=f"{exported_name}_collision")
                ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
                collision_geometry = ET.SubElement(collision, "geometry")
                ET.SubElement(collision_geometry, "mesh", filename=uri, scale="1 1 1")
            if include_inertial and len(vertices):
                _add_inertial(link_node, vertices, density)
            mesh_records[original_name] = (vertices, triangles, mesh_file)

    for joint in project.joints:
        node = ET.SubElement(
            robot_node,
            "joint",
            name=joint_names[joint.name],
            type=joint.type,
        )
        ET.SubElement(node, "parent", link=link_names[joint.parent])
        ET.SubElement(node, "child", link=link_names[joint.child])
        ET.SubElement(
            node,
            "origin",
            xyz=_format(joint.origin_xyz),
            rpy=_format(joint.origin_rpy),
        )
        if joint.type not in {"fixed", "floating"}:
            ET.SubElement(node, "axis", xyz=_format(joint.axis))
        if joint.type in {"revolute", "prismatic", "continuous"}:
            attributes = {
                "effort": f"{joint.effort:.12g}",
                "velocity": f"{joint.velocity:.12g}",
            }
            if joint.type != "continuous":
                attributes["lower"] = f"{joint.lower:.12g}"
                attributes["upper"] = f"{joint.upper:.12g}"
            ET.SubElement(node, "limit", **attributes)

    ET.indent(robot_node, space="  ")
    if urdf_filename is None:
        urdf_filename = safe_robot + ".urdf"
    if not urdf_filename.lower().endswith(".urdf"):
        urdf_filename += ".urdf"
    urdf_path = urdf_dir / Path(urdf_filename).name
    ET.ElementTree(robot_node).write(urdf_path, encoding="utf-8", xml_declaration=True)

    package_node = ET.Element("package", format="3")
    ET.SubElement(package_node, "name").text = safe_package
    ET.SubElement(package_node, "version").text = "0.0.1"
    ET.SubElement(package_node, "description").text = f"Meshes and URDF for {safe_robot}"
    ET.SubElement(package_node, "maintainer", email="user@example.com").text = "URDF Maker User"
    ET.SubElement(package_node, "license").text = "Proprietary"
    ET.SubElement(package_node, "exec_depend").text = "urdf"
    ET.indent(package_node, space="  ")
    ET.ElementTree(package_node).write(
        package_dir / "package.xml", encoding="utf-8", xml_declaration=True
    )
    (package_dir / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.8)\n"
        f"project({safe_package})\n"
        "find_package(ament_cmake REQUIRED)\n"
        "install(DIRECTORY urdf meshes DESTINATION share/${PROJECT_NAME})\n"
        "ament_package()\n",
        encoding="utf-8",
    )
    return urdf_path


# Explicit aliases keep the API discoverable for both "load/save" and
# "import/export" terminology used by desktop applications.
import_urdf = load_urdf
save_urdf = export_urdf
write_urdf = export_urdf


__all__ = [
    "MeshFormatError",
    "export_urdf",
    "import_urdf",
    "load_mesh",
    "load_urdf",
    "resolve_mesh_path",
    "save_urdf",
    "write_urdf",
]
