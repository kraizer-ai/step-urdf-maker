"""Triangle-mesh self-collision checks with a cached bounding-volume hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkTriangle
from vtkmodules.vtkFiltersCore import vtkCleanPolyData, vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersGeneral import vtkSampleImplicitFunctionFilter
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints
from vtkmodules.util.numpy_support import (
    numpy_to_vtk,
    numpy_to_vtkIdTypeArray,
    vtk_to_numpy,
)

from .model import (
    CollisionCandidate,
    CollisionSweepFinding,
    JointSpec,
    RobotProject,
    _oriented_boxes_overlap,
)


@dataclass(slots=True)
class _BvhNode:
    lower: np.ndarray
    upper: np.ndarray
    count: int
    indices: np.ndarray | None = None
    left: "_BvhNode | None" = None
    right: "_BvhNode | None" = None

    @property
    def leaf(self) -> bool:
        return self.indices is not None


@dataclass(slots=True)
class _TriangleMesh:
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_lower: np.ndarray
    triangle_upper: np.ndarray
    root: _BvhNode
    polydata: vtkPolyData
    sample_points: np.ndarray
    closed: bool
    implicit_distance: vtkImplicitPolyDataDistance

    @classmethod
    def from_part(cls, vertices: np.ndarray, triangles: np.ndarray) -> "_TriangleMesh":
        points = np.ascontiguousarray(vertices, dtype=np.float64)
        cells = np.ascontiguousarray(triangles, dtype=np.int64)
        triangle_points = points[cells]
        lower = triangle_points.min(axis=1)
        upper = triangle_points.max(axis=1)
        centers = (lower + upper) * 0.5
        indices = np.arange(len(cells), dtype=np.int64)
        root = _build_bvh(indices, lower, upper, centers)
        raw_polydata = _polydata(points, cells)
        cleaner = vtkCleanPolyData()
        cleaner.SetInputData(raw_polydata)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(1.0e-9)
        cleaner.PointMergingOn()
        cleaner.Update()
        polydata = vtkPolyData()
        polydata.DeepCopy(cleaner.GetOutput())
        implicit_distance = vtkImplicitPolyDataDistance()
        implicit_distance.SetInput(polydata)
        return cls(
            points,
            cells,
            lower,
            upper,
            root,
            polydata,
            np.ascontiguousarray(
                np.vstack((points, triangle_points.mean(axis=1))),
                dtype=np.float64,
            ),
            bool(vtkSelectEnclosedPoints.IsSurfaceClosed(polydata)),
            implicit_distance,
        )


def _polydata(vertices: np.ndarray, triangles: np.ndarray) -> vtkPolyData:
    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices), deep=True))
    cells = vtkCellArray()
    offsets = np.arange(0, len(triangles) * 3 + 1, 3, dtype=np.int64)
    cells.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(
            np.ascontiguousarray(triangles).reshape(-1), deep=True
        ),
    )
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)
    return polydata


def _point_polydata(points_array: np.ndarray) -> vtkPolyData:
    points = vtkPoints()
    points.SetData(
        numpy_to_vtk(np.ascontiguousarray(points_array, dtype=np.float64), deep=True)
    )
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    return polydata


def _build_bvh(
    indices: np.ndarray,
    triangle_lower: np.ndarray,
    triangle_upper: np.ndarray,
    centers: np.ndarray,
    *,
    leaf_size: int = 24,
) -> _BvhNode:
    lower = triangle_lower[indices].min(axis=0)
    upper = triangle_upper[indices].max(axis=0)
    if len(indices) <= leaf_size:
        return _BvhNode(lower, upper, len(indices), indices=indices.copy())
    center_bounds = np.ptp(centers[indices], axis=0)
    axis = int(np.argmax(center_bounds))
    order = indices[np.argsort(centers[indices, axis], kind="stable")]
    midpoint = len(order) // 2
    if midpoint <= 0 or midpoint >= len(order):
        return _BvhNode(lower, upper, len(indices), indices=indices.copy())
    return _BvhNode(
        lower,
        upper,
        len(indices),
        left=_build_bvh(
            order[:midpoint], triangle_lower, triangle_upper, centers,
            leaf_size=leaf_size,
        ),
        right=_build_bvh(
            order[midpoint:], triangle_lower, triangle_upper, centers,
            leaf_size=leaf_size,
        ),
    )


def _nodes_overlap(
    node_a: _BvhNode,
    node_b: _BvhNode,
    relative_b_to_a: np.ndarray,
    contact_tolerance: float,
) -> bool:
    center_a = (node_a.lower + node_a.upper) * 0.5
    half_a = (node_a.upper - node_a.lower) * 0.5
    center_b_zero = (node_b.lower + node_b.upper) * 0.5
    half_b = (node_b.upper - node_b.lower) * 0.5
    rotation = relative_b_to_a[:3, :3]
    center_b = rotation @ center_b_zero + relative_b_to_a[:3, 3]
    return _oriented_boxes_overlap(
        center_a,
        np.eye(3, dtype=float),
        half_a,
        center_b,
        rotation,
        half_b,
        contact_tolerance=contact_tolerance,
    )


def _meshes_intersect(
    mesh_a: _TriangleMesh,
    mesh_b: _TriangleMesh,
    relative_b_to_a: np.ndarray,
    *,
    contact_tolerance: float,
) -> bool:
    """Return whether any triangles intersect after mapping B into A's frame."""

    transformed_b = (
        mesh_b.vertices @ relative_b_to_a[:3, :3].T
        + relative_b_to_a[:3, 3]
    )
    stack: list[tuple[_BvhNode, _BvhNode]] = [(mesh_a.root, mesh_b.root)]
    while stack:
        node_a, node_b = stack.pop()
        if not _nodes_overlap(
            node_a, node_b, relative_b_to_a, contact_tolerance
        ):
            continue
        if node_a.leaf and node_b.leaf:
            assert node_a.indices is not None and node_b.indices is not None
            triangles_a = mesh_a.triangles[node_a.indices]
            triangles_b = mesh_b.triangles[node_b.indices]
            points_a = mesh_a.vertices[triangles_a]
            points_b = transformed_b[triangles_b]
            lower_a = points_a.min(axis=1)
            upper_a = points_a.max(axis=1)
            lower_b = points_b.min(axis=1)
            upper_b = points_b.max(axis=1)
            overlaps = np.all(
                lower_a[:, None, :] <= upper_b[None, :, :] + contact_tolerance,
                axis=2,
            ) & np.all(
                lower_b[None, :, :] <= upper_a[:, None, :] + contact_tolerance,
                axis=2,
            )
            for index_a, index_b in np.argwhere(overlaps):
                triangle_a = points_a[index_a]
                triangle_b = points_b[index_b]
                if vtkTriangle.TrianglesIntersect(
                    triangle_a[0],
                    triangle_a[1],
                    triangle_a[2],
                    triangle_b[0],
                    triangle_b[1],
                    triangle_b[2],
                ) and _triangles_cross_with_penetration(
                    triangle_a,
                    triangle_b,
                    contact_tolerance,
                ):
                    return True
            continue
        if node_b.leaf or (not node_a.leaf and node_a.count >= node_b.count):
            assert node_a.left is not None and node_a.right is not None
            stack.append((node_a.left, node_b))
            stack.append((node_a.right, node_b))
        else:
            assert node_b.left is not None and node_b.right is not None
            stack.append((node_a, node_b.left))
            stack.append((node_a, node_b.right))
    return False


def _triangles_cross_with_penetration(
    triangle_a: np.ndarray,
    triangle_b: np.ndarray,
    tolerance: float,
) -> bool:
    """Reject coplanar and tangent contact while retaining surface crossings."""

    normal_a = np.cross(
        triangle_a[1] - triangle_a[0],
        triangle_a[2] - triangle_a[0],
    )
    normal_b = np.cross(
        triangle_b[1] - triangle_b[0],
        triangle_b[2] - triangle_b[0],
    )
    norm_a = float(np.linalg.norm(normal_a))
    norm_b = float(np.linalg.norm(normal_b))
    if norm_a <= 1.0e-15 or norm_b <= 1.0e-15:
        return False
    distances_b = (triangle_b - triangle_a[0]) @ (normal_a / norm_a)
    distances_a = (triangle_a - triangle_b[0]) @ (normal_b / norm_b)
    return bool(
        distances_b.min() < -tolerance
        and distances_b.max() > tolerance
        and distances_a.min() < -tolerance
        and distances_a.max() > tolerance
    )


def _sample_points_penetrate_closed_mesh(
    query: _TriangleMesh,
    target: _TriangleMesh,
    relative_query_to_target: np.ndarray,
    tolerance: float,
) -> bool:
    transformed = (
        query.sample_points @ relative_query_to_target[:3, :3].T
        + relative_query_to_target[:3, 3]
    )
    query_polydata = _point_polydata(transformed)
    enclosed = vtkSelectEnclosedPoints()
    enclosed.SetInputData(query_polydata)
    enclosed.SetSurfaceData(target.polydata)
    enclosed.CheckSurfaceOff()
    enclosed.SetTolerance(1.0e-7)
    enclosed.Update()
    selected_array = enclosed.GetOutput().GetPointData().GetArray(
        "SelectedPoints"
    )
    if selected_array is None:
        return False
    selected = vtk_to_numpy(selected_array).astype(bool, copy=False)
    if not np.any(selected):
        return False

    # vtkSelectEnclosedPoints can classify points exactly on a surface as
    # inside.  Require real depth beyond the requested metric tolerance so
    # mating faces and tangent contact are not reported as penetration.
    selected_polydata = _point_polydata(transformed[selected])
    sampler = vtkSampleImplicitFunctionFilter()
    sampler.SetInputData(selected_polydata)
    sampler.SetImplicitFunction(target.implicit_distance)
    sampler.ComputeGradientsOff()
    sampler.Update()
    distances = sampler.GetOutput().GetPointData().GetScalars()
    if distances is None:
        return False
    return bool(np.any(np.abs(vtk_to_numpy(distances)) > tolerance))


def _closed_meshes_penetrate(
    mesh_a: _TriangleMesh,
    mesh_b: _TriangleMesh,
    relative_b_to_a: np.ndarray,
    tolerance: float,
) -> bool:
    if mesh_a.closed and _sample_points_penetrate_closed_mesh(
        mesh_b,
        mesh_a,
        relative_b_to_a,
        tolerance,
    ):
        return True
    if mesh_b.closed and _sample_points_penetrate_closed_mesh(
        mesh_a,
        mesh_b,
        np.linalg.inv(relative_b_to_a),
        tolerance,
    ):
        return True
    return False


class MeshCollisionChecker:
    """Run exact triangle-surface checks after the model's OBB broad phase."""

    def __init__(self, project: RobotProject):
        self.project = project
        self._meshes: dict[str, _TriangleMesh | None] = {}

    def _mesh(self, part_id: str) -> _TriangleMesh | None:
        if part_id not in self._meshes:
            part = self.project.parts[part_id]
            self._meshes[part_id] = (
                _TriangleMesh.from_part(part.vertices_zero, part.triangles)
                if len(part.triangles)
                else None
            )
        return self._meshes[part_id]

    def _link_deltas(
        self, positions: Mapping[str, float] | None
    ) -> dict[str, np.ndarray]:
        zero_fk = self.project.forward_kinematics(zero=True)
        current_fk = self.project.forward_kinematics(positions)
        return {
            name: current_fk[name] @ np.linalg.inv(zero_fk[name])
            for name in current_fk
        }

    def _candidate_intersects(
        self,
        candidate: CollisionCandidate,
        deltas: Mapping[str, np.ndarray],
        *,
        contact_tolerance: float,
    ) -> bool:
        mesh_a = self._mesh(candidate.part_a)
        mesh_b = self._mesh(candidate.part_b)
        if mesh_a is None or mesh_b is None:
            return False
        delta_a = deltas[candidate.link_a]
        delta_b = deltas[candidate.link_b]
        relative = np.linalg.inv(delta_a) @ delta_b
        if mesh_a.closed or mesh_b.closed:
            return _closed_meshes_penetrate(
                mesh_a,
                mesh_b,
                relative,
                contact_tolerance,
            )
        # Open triangle soups have no meaningful inside volume. Fall back to
        # strict surface crossing while still rejecting coplanar contact.
        if len(mesh_a.triangles) < len(mesh_b.triangles):
            mesh_a, mesh_b = mesh_b, mesh_a
            relative = np.linalg.inv(relative)
        return _meshes_intersect(
            mesh_a,
            mesh_b,
            relative,
            contact_tolerance=contact_tolerance,
        )

    def collisions_at(
        self,
        positions: Mapping[str, float] | None = None,
        *,
        contact_tolerance: float = 1.0e-9,
    ) -> list[CollisionCandidate]:
        """Return actual triangle-surface intersections at one pose."""

        tolerance = max(float(contact_tolerance), 0.0)
        broad = self.project.self_collision_candidates(
            positions, contact_tolerance=tolerance
        )
        if not broad:
            return []
        deltas = self._link_deltas(positions)
        return [
            candidate
            for candidate in broad
            if self._candidate_intersects(
                candidate, deltas, contact_tolerance=tolerance
            )
        ]

    def sampled_self_collisions(
        self,
        *,
        samples_per_joint: int = 3,
        max_joints: int = 32,
        contact_tolerance: float = 1.0e-9,
    ) -> tuple[list[CollisionCandidate], list[CollisionSweepFinding], int]:
        """Check current and sampled joint poses using exact mesh intersections."""

        sample_count = max(int(samples_per_joint), 2)
        joint_limit = max(int(max_joints), 0)
        current = self.collisions_at(contact_tolerance=contact_tolerance)
        current_keys = {
            frozenset((candidate.part_a, candidate.part_b)) for candidate in current
        }
        movable: list[tuple[JointSpec, tuple[float, float]]] = []
        for joint in self.project.joints:
            if joint.mimic_joint:
                continue
            limits = self.project._joint_preview_limits(joint)
            if limits is None or math.isclose(limits[0], limits[1]):
                continue
            movable.append((joint, limits))
        omitted = max(0, len(movable) - joint_limit)
        movable = movable[:joint_limit]
        base_positions = {
            joint.name: float(joint.position) for joint in self.project.joints
        }
        findings: list[CollisionSweepFinding] = []
        reported: set[tuple[frozenset[str], str]] = set()
        for joint, (lower, upper) in movable:
            for position in np.linspace(lower, upper, sample_count):
                if math.isclose(float(position), float(joint.position)):
                    continue
                positions = dict(base_positions)
                positions[joint.name] = float(position)
                for candidate in self.collisions_at(
                    positions,
                    contact_tolerance=contact_tolerance,
                ):
                    part_key = frozenset((candidate.part_a, candidate.part_b))
                    report_key = (part_key, joint.name)
                    if part_key in current_keys or report_key in reported:
                        continue
                    reported.add(report_key)
                    findings.append(
                        CollisionSweepFinding(
                            candidate,
                            joint.name,
                            float(position),
                        )
                    )
        if len(movable) > 1:
            for ratio in np.linspace(0.0, 1.0, sample_count):
                positions = dict(base_positions)
                for joint, (lower, upper) in movable:
                    positions[joint.name] = float(
                        lower + (upper - lower) * ratio
                    )
                if all(
                    math.isclose(positions[joint.name], joint.position)
                    for joint, _limits in movable
                ):
                    continue
                for candidate in self.collisions_at(
                    positions,
                    contact_tolerance=contact_tolerance,
                ):
                    part_key = frozenset((candidate.part_a, candidate.part_b))
                    report_key = (part_key, "__combined__")
                    if part_key in current_keys or report_key in reported:
                        continue
                    reported.add(report_key)
                    findings.append(
                        CollisionSweepFinding(candidate, None, float(ratio))
                    )
        return current, findings, omitted


__all__ = ["MeshCollisionChecker"]
