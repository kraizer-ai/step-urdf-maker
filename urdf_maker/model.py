"""Core, UI-independent data model for the URDF maker.

The central convention in this module is that :class:`ScenePart.vertices_zero`
contains metres in the project's world frame at the robot's zero pose.  This
makes CAD selections straightforward (CAD importers normally provide assembled
coordinates), while :meth:`RobotProject.link_vertices_local` converts the data
back to URDF link-local coordinates for export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


VALID_JOINT_TYPES = frozenset(
    {"fixed", "revolute", "continuous", "prismatic", "floating", "planar"}
)
_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


class ProjectValidationError(ValueError):
    """Raised when a project cannot form a valid URDF kinematic tree."""

    def __init__(self, errors: Sequence[str]):
        self.errors = list(errors)
        super().__init__("Invalid robot project:\n- " + "\n- ".join(self.errors))


def is_valid_name(name: str) -> bool:
    """Return whether *name* is safe as a URDF/ROS identifier."""

    return bool(name and _VALID_NAME.fullmatch(name))


def sanitize_name(name: str, prefix: str = "item") -> str:
    """Convert arbitrary display text to a stable URDF-safe identifier."""

    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())
    value = value.strip("_.-")
    if not value:
        value = prefix
    if not (value[0].isalpha() or value[0] == "_"):
        value = f"{prefix}_{value}"
    return value


def _array(value: Sequence[float] | np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {result.shape}")
    return result.copy()


def rpy_matrix(rpy: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the URDF fixed-axis roll/pitch/yaw rotation matrix.

    URDF applies roll about X, then pitch about Y, then yaw about Z; for column
    vectors the resulting matrix is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
    """

    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1.0, 0.0, 0.0), (0.0, cr, -sr), (0.0, sr, cr)))
    ry = np.array(((cp, 0.0, sp), (0.0, 1.0, 0.0), (-sp, 0.0, cp)))
    rz = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    return rz @ ry @ rx


def transform_matrix(
    xyz: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
    rpy: Sequence[float] | np.ndarray = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Build a homogeneous transform from URDF xyz and rpy values."""

    result = np.eye(4, dtype=float)
    result[:3, :3] = rpy_matrix(rpy)
    result[:3, 3] = np.asarray(xyz, dtype=float)
    return result


def matrix_rpy(rotation: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Recover URDF roll/pitch/yaw from a 3x3 rotation matrix."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    horizontal = math.hypot(matrix[0, 0], matrix[1, 0])
    pitch = math.atan2(-matrix[2, 0], horizontal)
    if horizontal > 1e-12:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:  # Gimbal lock: choose roll=0 and retain the observable combination.
        roll = 0.0
        yaw = math.atan2(-matrix[0, 1], matrix[1, 1])
    return np.array((roll, pitch, yaw), dtype=float)


def axis_angle_matrix(axis: Sequence[float] | np.ndarray, angle: float) -> np.ndarray:
    """Return a homogeneous rotation about *axis* using Rodrigues' formula."""

    vector = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-15:
        raise ValueError("A moving joint axis must be non-zero")
    x, y, z = vector / norm
    c, s = math.cos(float(angle)), math.sin(float(angle))
    one_c = 1.0 - c
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.array(
        (
            (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
            (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
            (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
        )
    )
    return result


def apply_transform(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to an ``(N, 3)`` vertex array."""

    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("matrix must have shape (4, 4)")
    return points @ transform[:3, :3].T + transform[:3, 3]


def _oriented_boxes_overlap(
    center_a: np.ndarray,
    axes_a: np.ndarray,
    half_a: np.ndarray,
    center_b: np.ndarray,
    axes_b: np.ndarray,
    half_b: np.ndarray,
    *,
    contact_tolerance: float,
) -> bool:
    """Test two oriented boxes with the separating-axis theorem."""

    axes: list[np.ndarray] = [
        axes_a[:, index] for index in range(3)
    ] + [axes_b[:, index] for index in range(3)]
    for axis_a in (axes_a[:, index] for index in range(3)):
        for axis_b in (axes_b[:, index] for index in range(3)):
            cross = np.cross(axis_a, axis_b)
            norm = float(np.linalg.norm(cross))
            if norm > 1.0e-10:
                axes.append(cross / norm)

    center_delta = center_b - center_a
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-12:
            continue
        direction = axis / norm
        radius_a = float(np.sum(half_a * np.abs(axes_a.T @ direction)))
        radius_b = float(np.sum(half_b * np.abs(axes_b.T @ direction)))
        separation = abs(float(np.dot(center_delta, direction)))
        # Mere face/edge contact is allowed.  Every separating axis must have
        # positive penetration beyond the small numerical tolerance.
        if separation >= radius_a + radius_b - contact_tolerance:
            return False
    return True


@dataclass
class ScenePart:
    """One selectable triangle mesh in the zero-pose world frame."""

    id: str
    name: str
    vertices_zero: np.ndarray
    triangles: np.ndarray
    color: Sequence[float] | np.ndarray = (0.72, 0.74, 0.78, 1.0)
    link_name: str | None = None
    visible: bool = True
    feature_axes: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.name = str(self.name)
        vertices = np.asarray(self.vertices_zero, dtype=float)
        if vertices.size == 0:
            vertices = np.empty((0, 3), dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices_zero must have shape (N, 3)")
        triangles = np.asarray(self.triangles, dtype=np.int64)
        if triangles.size == 0:
            triangles = np.empty((0, 3), dtype=np.int64)
        if triangles.ndim != 2 or triangles.shape[1] != 3:
            raise ValueError("triangles must have shape (M, 3)")
        if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
            raise ValueError("triangle index is outside vertices_zero")
        rgba = np.asarray(self.color, dtype=float).reshape(-1)
        if rgba.size == 3:
            rgba = np.append(rgba, 1.0)
        if rgba.size != 4:
            raise ValueError("color must contain RGB or RGBA values")
        self.vertices_zero = vertices.copy()
        self.triangles = triangles.copy()
        self.color = np.clip(rgba, 0.0, 1.0)
        self.link_name = str(self.link_name) if self.link_name is not None else None
        self.visible = bool(self.visible)
        self.feature_axes = [dict(item) for item in self.feature_axes]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not len(self.vertices_zero):
            return None
        return self.vertices_zero.min(axis=0), self.vertices_zero.max(axis=0)


@dataclass(frozen=True)
class CollisionCandidate:
    """A conservative overlap between two rigid part bounding boxes."""

    link_a: str
    link_b: str
    part_a: str
    part_b: str


@dataclass(frozen=True)
class CollisionSweepFinding:
    """A new collision candidate found at one sampled joint position."""

    candidate: CollisionCandidate
    joint_name: str | None
    position: float


@dataclass
class LinkSpec:
    """A URDF link and the selectable parts rigidly attached to it."""

    name: str
    part_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.part_ids = list(dict.fromkeys(str(item) for item in self.part_ids))


@dataclass
class JointSpec:
    """A URDF joint, including the current preview position."""

    name: str
    type: str
    parent: str
    child: str
    origin_xyz: Sequence[float] | np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    origin_rpy: Sequence[float] | np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    axis: Sequence[float] | np.ndarray = field(
        default_factory=lambda: np.array((1.0, 0.0, 0.0), dtype=float)
    )
    lower: float | None = None
    upper: float | None = None
    effort: float = 100.0
    velocity: float = 1.0
    damping: float = 0.0
    friction: float = 0.0
    position: float = 0.0
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0
    mimic_auto: bool = False
    mimic_reverse: bool = False
    drive_source_joint: str | None = None
    drive_max_velocity: float = 2.0 * math.pi
    drive_deadband: float = 0.03
    drive_reverse: bool = False

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.type = str(self.type).lower()
        self.parent = str(self.parent)
        self.child = str(self.child)
        self.origin_xyz = _array(self.origin_xyz, (3,), "origin_xyz")
        self.origin_rpy = _array(self.origin_rpy, (3,), "origin_rpy")
        self.axis = _array(self.axis, (3,), "axis")
        if self.type == "fixed":
            self.lower, self.upper, self.position = 0.0, 0.0, 0.0
        elif self.type == "revolute" and self.lower is None and self.upper is None:
            self.lower, self.upper = -math.pi, math.pi
        elif self.type == "prismatic" and self.lower is None and self.upper is None:
            self.lower, self.upper = -0.1, 0.1
        elif self.type == "continuous":
            # Continuous joints are unbounded in URDF; these values are only a
            # convenient one-turn preview range for the desktop slider.
            self.lower = -math.pi if self.lower is None else self.lower
            self.upper = math.pi if self.upper is None else self.upper
        self.lower = float(self.lower) if self.lower is not None else None
        self.upper = float(self.upper) if self.upper is not None else None
        self.effort = float(self.effort)
        self.velocity = float(self.velocity)
        self.damping = float(self.damping)
        self.friction = float(self.friction)
        self.position = float(self.position)
        self.mimic_joint = (
            str(self.mimic_joint).strip() if self.mimic_joint else None
        )
        self.mimic_multiplier = float(self.mimic_multiplier)
        self.mimic_offset = float(self.mimic_offset)
        self.mimic_auto = bool(self.mimic_auto and self.mimic_joint)
        self.mimic_reverse = bool(self.mimic_reverse and self.mimic_joint)
        self.drive_source_joint = (
            str(self.drive_source_joint).strip() if self.drive_source_joint else None
        )
        self.drive_max_velocity = float(self.drive_max_velocity)
        self.drive_deadband = float(self.drive_deadband)
        self.drive_reverse = bool(self.drive_reverse and self.drive_source_joint)

    def origin_transform(self) -> np.ndarray:
        return transform_matrix(self.origin_xyz, self.origin_rpy)

    def motion_transform(self, position: float | None = None) -> np.ndarray:
        value = self.position if position is None else float(position)
        if self.type in {"fixed"}:
            return np.eye(4, dtype=float)
        if self.type in {"revolute", "continuous"}:
            return axis_angle_matrix(self.axis, value)
        if self.type == "prismatic":
            norm = float(np.linalg.norm(self.axis))
            if norm <= 1e-15:
                raise ValueError(f"Joint {self.name!r} has a zero axis")
            result = np.eye(4, dtype=float)
            result[:3, 3] = self.axis / norm * value
            return result
        # A scalar preview cannot describe planar/floating joints.  Keeping the
        # imported zero transform is more useful than refusing to display them.
        if self.type in {"planar", "floating"}:
            return np.eye(4, dtype=float)
        raise ValueError(f"Unsupported joint type: {self.type}")

    def transform(self, position: float | None = None) -> np.ndarray:
        """Return the parent-to-child transform at *position*."""

        return self.origin_transform() @ self.motion_transform(position)

    def clamp(self, value: float) -> float:
        value = float(value)
        if self.type == "continuous":
            return value
        if self.lower is not None:
            value = max(self.lower, value)
        if self.upper is not None:
            value = min(self.upper, value)
        return value


@dataclass
class RobotProject:
    """Complete editable robot scene and kinematic tree."""

    name: str
    parts: Mapping[str, ScenePart] | Iterable[ScenePart] = field(default_factory=dict)
    links: Mapping[str, LinkSpec] | Iterable[LinkSpec] = field(default_factory=dict)
    joints: Iterable[JointSpec] = field(default_factory=list)
    root_link: str | None = None
    source_path: str | None = None
    source_kind: str | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if isinstance(self.parts, Mapping):
            part_values = list(self.parts.values())
        else:
            part_values = list(self.parts)
        if isinstance(self.links, Mapping):
            link_values = list(self.links.values())
        else:
            link_values = list(self.links)
        part_identifiers = [part.id for part in part_values]
        link_identifiers = [link.name for link in link_values]
        self._duplicate_part_ids = sorted(
            {item for item in part_identifiers if part_identifiers.count(item) > 1}
        )
        self._duplicate_link_names = sorted(
            {item for item in link_identifiers if link_identifiers.count(item) > 1}
        )
        self.parts = {part.id: part for part in part_values}
        self.links = {link.name: link for link in link_values}
        self.joints = list(self.joints)
        self.root_link = str(self.root_link) if self.root_link is not None else None
        self.source_path = str(self.source_path) if self.source_path is not None else None
        self.source_kind = str(self.source_kind) if self.source_kind is not None else None
        self.metadata = dict(self.metadata)

        # Reconcile the two convenient representations of assignment.  Explicit
        # LinkSpec.part_ids wins, then unlisted ScenePart.link_name is appended.
        assigned: set[str] = set()
        for link in self.links.values():
            for part_id in link.part_ids:
                if part_id in self.parts and part_id not in assigned:
                    self.parts[part_id].link_name = link.name
                    assigned.add(part_id)
        for part in self.parts.values():
            if part.id in assigned or part.link_name is None:
                continue
            link = self.links.get(part.link_name)
            if link is not None:
                link.part_ids.append(part.id)
                assigned.add(part.id)

        if self.root_link is None and self.links:
            roots = self.root_candidates()
            if len(roots) == 1:
                self.root_link = roots[0]

    def root_candidates(self) -> list[str]:
        children = {joint.child for joint in self.joints}
        return [name for name in self.links if name not in children]

    def joint(self, name: str) -> JointSpec:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise KeyError(name)

    def joint_for_child(self, child_link: str) -> JointSpec | None:
        for joint in self.joints:
            if joint.child == child_link:
                return joint
        return None

    def children_of(self, parent_link: str) -> list[JointSpec]:
        return [joint for joint in self.joints if joint.parent == parent_link]

    @staticmethod
    def _joint_preview_limits(joint: JointSpec) -> tuple[float, float] | None:
        if joint.type not in {"revolute", "continuous", "prismatic"}:
            return None
        lower = joint.lower
        upper = joint.upper
        if lower is None or upper is None:
            if joint.type == "continuous":
                return -math.pi, math.pi
            return None
        if not math.isfinite(lower) or not math.isfinite(upper):
            return None
        return float(lower), float(upper)

    def mimic_parameters(self, joint: JointSpec | str) -> tuple[float, float]:
        """Return the active linear mimic multiplier and offset.

        Auto mode maps the source joint's state 0/1 limits to the dependent
        joint's state 0/1 limits.  This keeps mixed revolute/prismatic units
        correct without asking the user to calculate metres per radian.
        """

        target = self.joint(joint) if isinstance(joint, str) else joint
        if not target.mimic_joint:
            return float(target.mimic_multiplier), float(target.mimic_offset)
        if target.mimic_auto:
            try:
                source = self.joint(target.mimic_joint)
            except KeyError:
                return float(target.mimic_multiplier), float(target.mimic_offset)
            source_limits = self._joint_preview_limits(source)
            target_limits = self._joint_preview_limits(target)
            if source_limits is not None and target_limits is not None:
                source_0, source_1 = source_limits
                target_0, target_1 = target_limits
                if target.mimic_reverse:
                    target_0, target_1 = target_1, target_0
                source_range = source_1 - source_0
                if abs(source_range) > 1.0e-15:
                    multiplier = (target_1 - target_0) / source_range
                    offset = target_0 - multiplier * source_0
                    return float(multiplier), float(offset)
        return float(target.mimic_multiplier), float(target.mimic_offset)

    def resolved_joint_positions(
        self,
        positions: Mapping[str, float] | None = None,
        *,
        zero: bool = False,
    ) -> dict[str, float]:
        """Resolve independent and chained mimic joint preview positions."""

        by_name = {joint.name: joint for joint in self.joints}
        result: dict[str, float] = {}
        active: set[str] = set()

        def resolve(name: str) -> float:
            if name in result:
                return result[name]
            if name in active:
                raise ProjectValidationError(
                    [f"Mimic joint dependency contains a cycle at {name!r}"]
                )
            joint = by_name[name]
            if zero:
                value = 0.0
            elif joint.mimic_joint and joint.mimic_joint in by_name:
                active.add(name)
                source_value = resolve(joint.mimic_joint)
                active.remove(name)
                multiplier, offset = self.mimic_parameters(joint)
                value = joint.clamp(multiplier * source_value + offset)
            elif positions is not None and name in positions:
                value = joint.clamp(float(positions[name]))
            else:
                value = joint.clamp(joint.position)
            result[name] = float(value)
            return result[name]

        for joint_name in by_name:
            resolve(joint_name)
        return result

    def apply_mimic_positions(self) -> dict[str, float]:
        """Update dependent preview values and return every resolved value."""

        resolved = self.resolved_joint_positions()
        for joint in self.joints:
            if joint.mimic_joint:
                joint.position = resolved[joint.name]
        return resolved

    def drive_fraction(
        self,
        joint: JointSpec | str,
        positions: Mapping[str, float] | None = None,
    ) -> float:
        """Return a wheel drive command in the range -1..1.

        The source lever's lower limit is full reverse, the midpoint is
        neutral, and the upper limit is full forward.  A small configurable
        deadband makes it easy to stop exactly at the centre of a UI slider.
        """

        target = self.joint(joint) if isinstance(joint, str) else joint
        if not target.drive_source_joint:
            return 0.0
        try:
            source = self.joint(target.drive_source_joint)
        except KeyError:
            return 0.0
        limits = self._joint_preview_limits(source)
        if limits is None:
            return 0.0
        lower, upper = limits
        span = upper - lower
        if not math.isfinite(span) or abs(span) <= 1.0e-15:
            return 0.0
        resolved = self.resolved_joint_positions(positions)
        value = resolved.get(source.name, source.position)
        fraction = 2.0 * (float(value) - lower) / span - 1.0
        fraction = min(max(fraction, -1.0), 1.0)
        deadband = min(max(float(target.drive_deadband), 0.0), 0.999999)
        magnitude = abs(fraction)
        if magnitude <= deadband:
            fraction = 0.0
        else:
            fraction = math.copysign(
                (magnitude - deadband) / (1.0 - deadband), fraction
            )
        if target.drive_reverse:
            fraction = -fraction
        return float(fraction)

    def drive_velocity(
        self,
        joint: JointSpec | str,
        positions: Mapping[str, float] | None = None,
    ) -> float:
        """Return the configured continuous-joint angular velocity in rad/s."""

        target = self.joint(joint) if isinstance(joint, str) else joint
        return self.drive_fraction(target, positions) * float(target.drive_max_velocity)

    def create_link(self, name: str, part_ids: Iterable[str] = ()) -> LinkSpec:
        """Create a link and optionally move parts into it."""

        name = str(name)
        if name in self.links:
            raise ValueError(f"Link {name!r} already exists")
        identifiers = list(dict.fromkeys(str(item) for item in part_ids))
        missing = [item for item in identifiers if item not in self.parts]
        if missing:
            raise KeyError(f"Unknown part(s): {', '.join(missing)}")
        link = LinkSpec(name)
        self.links[name] = link
        self.assign_parts(identifiers, name)
        if self.root_link is None:
            self.root_link = name
        return link

    def assign_parts(self, part_ids: Iterable[str], link_name: str | None) -> None:
        """Atomically reassign parts, removing their previous link membership."""

        identifiers = list(dict.fromkeys(str(item) for item in part_ids))
        missing = [item for item in identifiers if item not in self.parts]
        if missing:
            raise KeyError(f"Unknown part(s): {', '.join(missing)}")
        if link_name is not None and link_name not in self.links:
            raise KeyError(f"Unknown link: {link_name}")
        selected = set(identifiers)
        for link in self.links.values():
            link.part_ids = [item for item in link.part_ids if item not in selected]
        if link_name is not None:
            self.links[link_name].part_ids.extend(identifiers)
        for identifier in identifiers:
            self.parts[identifier].link_name = link_name

    def merge_links(self, target: str, sources: str | Iterable[str]) -> LinkSpec:
        """Merge rigid links and redirect surrounding joints to *target*.

        Joints whose two ends become the same link are removed.  The method does
        not silently resolve a genuinely ambiguous multi-parent result; callers
        can inspect :meth:`validate` after a merge.
        """

        if target not in self.links:
            raise KeyError(f"Unknown target link: {target}")
        if isinstance(sources, str):
            source_names = [sources]
        else:
            source_names = list(sources)
        source_names = list(dict.fromkeys(name for name in source_names if name != target))
        missing = [name for name in source_names if name not in self.links]
        if missing:
            raise KeyError(f"Unknown source link(s): {', '.join(missing)}")
        source_set = set(source_names)
        cluster = source_set | {target}
        try:
            zero_fk = self.forward_kinematics(zero=True)
        except ProjectValidationError:
            zero_fk = None
        part_ids: list[str] = []
        for name in source_names:
            part_ids.extend(self.links[name].part_ids)
        self.assign_parts(part_ids, target)
        redirected: list[JointSpec] = []
        for joint in self.joints:
            old_parent, old_child = joint.parent, joint.child
            if old_parent in cluster and old_child in cluster:
                continue
            new_parent = target if old_parent in source_set else old_parent
            new_child = target if old_child in source_set else old_child
            if zero_fk is not None:
                parent_world = zero_fk[target] if new_parent == target else zero_fk[old_parent]
                child_world = zero_fk[target] if new_child == target else zero_fk[old_child]
                relative = np.linalg.inv(parent_world) @ child_world
                joint.origin_xyz = relative[:3, 3].copy()
                joint.origin_rpy = matrix_rpy(relative[:3, :3])
                if old_child in source_set and joint.type in {
                    "revolute",
                    "continuous",
                    "prismatic",
                }:
                    # The axis was expressed in the old zero-pose joint/child
                    # frame; express the same world direction in target frame.
                    joint.axis = (
                        zero_fk[target][:3, :3].T
                        @ zero_fk[old_child][:3, :3]
                        @ joint.axis
                    )
            joint.parent, joint.child = new_parent, new_child
            if new_parent != new_child:
                redirected.append(joint)
        self.joints = redirected
        remaining_joint_names = {joint.name for joint in self.joints}
        for joint in self.joints:
            if joint.mimic_joint and joint.mimic_joint not in remaining_joint_names:
                joint.mimic_joint = None
                joint.mimic_auto = False
                joint.mimic_reverse = False
            if (
                joint.drive_source_joint
                and joint.drive_source_joint not in remaining_joint_names
            ):
                joint.drive_source_joint = None
                joint.drive_reverse = False
        for name in source_names:
            del self.links[name]
        if self.root_link in source_set:
            self.root_link = target
        return self.links[target]

    def set_joint_position(self, name: str, position: float, clamp: bool = True) -> float:
        joint = self.joint(name)
        if joint.mimic_joint:
            return self.apply_mimic_positions()[joint.name]
        value = joint.clamp(position) if clamp else float(position)
        joint.position = value
        self.apply_mimic_positions()
        return value

    def nudge_joint(self, name: str, delta: float, clamp: bool = True) -> float:
        joint = self.joint(name)
        return self.set_joint_position(name, joint.position + float(delta), clamp=clamp)

    def forward_kinematics(
        self,
        positions: Mapping[str, float] | None = None,
        *,
        zero: bool = False,
    ) -> dict[str, np.ndarray]:
        """Calculate world transforms for every link.

        ``positions`` maps joint names to values and overrides current preview
        values.  ``zero=True`` forces all scalar joints to zero and is used when
        converting :attr:`ScenePart.vertices_zero` into local mesh coordinates.
        """

        if not self.links:
            return {}
        root = self.root_link
        if root not in self.links:
            roots = self.root_candidates()
            if len(roots) != 1:
                raise ProjectValidationError(
                    [f"Expected one root link, found {len(roots)}: {', '.join(roots)}"]
                )
            root = roots[0]
        by_parent: dict[str, list[JointSpec]] = {}
        parent_for_child: dict[str, str] = {}
        for joint in self.joints:
            if joint.parent not in self.links or joint.child not in self.links:
                raise ProjectValidationError(
                    [f"Joint {joint.name!r} references a missing parent or child link"]
                )
            if joint.child in parent_for_child:
                raise ProjectValidationError(
                    [f"Link {joint.child!r} has more than one parent joint"]
                )
            parent_for_child[joint.child] = joint.parent
            by_parent.setdefault(joint.parent, []).append(joint)

        resolved_positions = self.resolved_joint_positions(positions, zero=zero)
        transforms = {root: np.eye(4, dtype=float)}
        queue = [root]
        while queue:
            parent = queue.pop(0)
            for joint in by_parent.get(parent, ()):  # deterministic input order
                if joint.child in transforms:
                    raise ProjectValidationError([f"Kinematic cycle reaches {joint.child!r}"])
                value = resolved_positions[joint.name]
                transforms[joint.child] = transforms[parent] @ joint.transform(value)
                queue.append(joint.child)
        if len(transforms) != len(self.links):
            missing = [name for name in self.links if name not in transforms]
            raise ProjectValidationError(
                [f"Links are disconnected from root {root!r}: {', '.join(missing)}"]
            )
        return transforms

    # Short alias useful in render loops.
    fk = forward_kinematics

    def link_vertices_local(self, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return one combined zero-pose mesh expressed in link coordinates."""

        if link_name not in self.links:
            raise KeyError(link_name)
        zero_fk = self.forward_kinematics(zero=True)
        world_to_link = np.linalg.inv(zero_fk[link_name])
        vertices: list[np.ndarray] = []
        triangles: list[np.ndarray] = []
        offset = 0
        for part_id in self.links[link_name].part_ids:
            part = self.parts.get(part_id)
            if part is None:
                continue
            local = apply_transform(part.vertices_zero, world_to_link)
            vertices.append(local)
            triangles.append(part.triangles + offset)
            offset += len(local)
        if not vertices:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
        return np.vstack(vertices), np.vstack(triangles)

    def transformed_part_vertices(
        self,
        part_id: str,
        positions: Mapping[str, float] | None = None,
    ) -> np.ndarray:
        """Return a part's vertices after applying preview joint positions."""

        part = self.parts[part_id]
        if part.link_name is None or part.link_name not in self.links:
            return part.vertices_zero.copy()
        zero_fk = self.forward_kinematics(zero=True)
        current_fk = self.forward_kinematics(positions)
        delta = current_fk[part.link_name] @ np.linalg.inv(zero_fk[part.link_name])
        return apply_transform(part.vertices_zero, delta)

    def transformed_parts(
        self, positions: Mapping[str, float] | None = None, *, visible_only: bool = False
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        zero_fk = self.forward_kinematics(zero=True) if self.links else {}
        current_fk = self.forward_kinematics(positions) if self.links else {}
        deltas = {
            name: current_fk[name] @ np.linalg.inv(zero_fk[name]) for name in current_fk
        }
        for identifier, part in self.parts.items():
            if visible_only and not part.visible:
                continue
            if part.link_name in deltas:
                result[identifier] = apply_transform(part.vertices_zero, deltas[part.link_name])
            else:
                result[identifier] = part.vertices_zero.copy()
        return result

    def self_collision_candidates(
        self,
        positions: Mapping[str, float] | None = None,
        *,
        contact_tolerance: float = 1.0e-7,
    ) -> list[CollisionCandidate]:
        """Return conservative inter-link collisions at one robot pose.

        Each part is represented by its zero-pose axis-aligned bounds transformed
        as an oriented bounding box.  This is deliberately a fast broad-phase
        check: it can report a candidate for concave meshes that do not actually
        touch, but it will not stall interactive URDF loading on dense CAD meshes.
        Parts on the same rigid link are never compared.
        """

        bounds_cache = {
            part.id: part.bounds for part in self.parts.values()
        }
        return self._self_collision_candidates(
            positions,
            contact_tolerance=max(float(contact_tolerance), 0.0),
            bounds_cache=bounds_cache,
        )

    def _self_collision_candidates(
        self,
        positions: Mapping[str, float] | None,
        *,
        contact_tolerance: float,
        bounds_cache: Mapping[str, tuple[np.ndarray, np.ndarray] | None],
    ) -> list[CollisionCandidate]:
        if not self.links or not self.parts:
            return []
        zero_fk = self.forward_kinematics(zero=True)
        current_fk = self.forward_kinematics(positions)
        deltas = {
            name: current_fk[name] @ np.linalg.inv(zero_fk[name])
            for name in current_fk
        }
        boxes: list[
            tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for part in self.parts.values():
            link_name = part.link_name
            bounds = bounds_cache.get(part.id)
            if link_name not in deltas or bounds is None:
                continue
            lower, upper = bounds
            center_zero = (lower + upper) * 0.5
            half_size = np.maximum((upper - lower) * 0.5, 0.0)
            delta = deltas[link_name]
            center = delta[:3, :3] @ center_zero + delta[:3, 3]
            axes = delta[:3, :3]
            world_half = np.abs(axes) @ half_size
            boxes.append(
                (part.id, link_name, center, axes, half_size, world_half)
            )

        candidates: list[CollisionCandidate] = []
        for index, first in enumerate(boxes):
            part_a, link_a, center_a, axes_a, half_a, world_half_a = first
            for (
                part_b,
                link_b,
                center_b,
                axes_b,
                half_b,
                world_half_b,
            ) in boxes[index + 1 :]:
                if link_a == link_b:
                    continue
                if np.any(
                    np.abs(center_b - center_a)
                    >= world_half_a + world_half_b - contact_tolerance
                ):
                    continue
                if not _oriented_boxes_overlap(
                    center_a,
                    axes_a,
                    half_a,
                    center_b,
                    axes_b,
                    half_b,
                    contact_tolerance=contact_tolerance,
                ):
                    continue
                candidates.append(
                    CollisionCandidate(link_a, link_b, part_a, part_b)
                )
        return candidates

    def sampled_self_collision_candidates(
        self,
        *,
        samples_per_joint: int = 3,
        max_joints: int = 32,
        contact_tolerance: float = 1.0e-7,
    ) -> tuple[list[CollisionCandidate], list[CollisionSweepFinding], int]:
        """Check the current pose and samples across every scalar joint range.

        The return value is ``(current, motion, omitted_joint_count)``.  Motion
        findings contain only part pairs that were not already overlapping at
        the current pose, which suppresses the normal contact around assembled
        joints while still identifying newly introduced interference.
        """

        sample_count = max(int(samples_per_joint), 2)
        joint_limit = max(int(max_joints), 0)
        bounds_cache = {
            part.id: part.bounds for part in self.parts.values()
        }
        current = self._self_collision_candidates(
            None,
            contact_tolerance=contact_tolerance,
            bounds_cache=bounds_cache,
        )
        current_keys = {
            frozenset((candidate.part_a, candidate.part_b)) for candidate in current
        }
        movable: list[tuple[JointSpec, tuple[float, float]]] = []
        for joint in self.joints:
            if joint.mimic_joint:
                continue
            limits = self._joint_preview_limits(joint)
            if limits is None or math.isclose(limits[0], limits[1]):
                continue
            movable.append((joint, limits))
        omitted = max(0, len(movable) - joint_limit)
        movable = movable[:joint_limit]
        base_positions = {
            joint.name: float(joint.position) for joint in self.joints
        }
        findings: list[CollisionSweepFinding] = []
        reported: set[tuple[frozenset[str], str]] = set()
        for joint, (lower, upper) in movable:
            for position in np.linspace(lower, upper, sample_count):
                if math.isclose(float(position), float(joint.position)):
                    continue
                positions = dict(base_positions)
                positions[joint.name] = float(position)
                for candidate in self._self_collision_candidates(
                    positions,
                    contact_tolerance=contact_tolerance,
                    bounds_cache=bounds_cache,
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
        return current, findings, omitted

    def validate(
        self,
        *,
        raise_on_error: bool = False,
        check_names: bool = True,
    ) -> list[str]:
        """Return actionable tree, naming, assignment, and limit errors."""

        errors: list[str] = []
        for identifier in self._duplicate_part_ids:
            errors.append(f"Duplicate part id {identifier!r}")
        for name in self._duplicate_link_names:
            errors.append(f"Duplicate link name {name!r}")
        if check_names and not is_valid_name(self.name):
            errors.append(f"Robot name {self.name!r} is not URDF-safe")
        if not self.links:
            errors.append("The project has no links")
        if check_names:
            for name in self.links:
                if not is_valid_name(name):
                    errors.append(f"Link name {name!r} is not URDF-safe")
        if self.root_link not in self.links:
            errors.append(f"Root link {self.root_link!r} does not exist")

        joint_names: set[str] = set()
        children: dict[str, str] = {}
        adjacency: dict[str, list[str]] = {}
        for joint in self.joints:
            if check_names and not is_valid_name(joint.name):
                errors.append(f"Joint name {joint.name!r} is not URDF-safe")
            if joint.name in joint_names:
                errors.append(f"Duplicate joint name {joint.name!r}")
            joint_names.add(joint.name)
            if joint.type not in VALID_JOINT_TYPES:
                errors.append(f"Joint {joint.name!r} has unsupported type {joint.type!r}")
            if joint.parent not in self.links:
                errors.append(f"Joint {joint.name!r} parent {joint.parent!r} is missing")
            if joint.child not in self.links:
                errors.append(f"Joint {joint.name!r} child {joint.child!r} is missing")
            if joint.parent == joint.child:
                errors.append(f"Joint {joint.name!r} connects a link to itself")
            if joint.child in children:
                errors.append(
                    f"Link {joint.child!r} has multiple parents: "
                    f"{children[joint.child]!r} and {joint.name!r}"
                )
            children[joint.child] = joint.name
            adjacency.setdefault(joint.parent, []).append(joint.child)
            values = np.r_[joint.origin_xyz, joint.origin_rpy, joint.axis]
            if not np.all(np.isfinite(values)):
                errors.append(f"Joint {joint.name!r} contains non-finite coordinates")
            scalar_values = [
                joint.position,
                joint.effort,
                joint.velocity,
                joint.damping,
                joint.friction,
            ]
            scalar_values.extend(
                value for value in (joint.lower, joint.upper) if value is not None
            )
            finite_scalars = all(math.isfinite(value) for value in scalar_values)
            if not finite_scalars:
                errors.append(f"Joint {joint.name!r} contains non-finite limits or settings")
            if joint.type in {"revolute", "continuous", "prismatic"}:
                if np.linalg.norm(joint.axis) <= 1e-15:
                    errors.append(f"Joint {joint.name!r} has a zero axis")
            if joint.type in {"revolute", "prismatic"}:
                if joint.lower is None or joint.upper is None:
                    errors.append(f"Joint {joint.name!r} requires lower and upper limits")
                elif finite_scalars and joint.lower > joint.upper:
                    errors.append(f"Joint {joint.name!r} lower limit exceeds upper limit")
                elif finite_scalars and not (
                    joint.lower - 1e-12 <= joint.position <= joint.upper + 1e-12
                ):
                    errors.append(f"Joint {joint.name!r} preview position is outside its limits")
            if finite_scalars and (
                joint.effort < 0
                or joint.velocity < 0
                or joint.damping < 0
                or joint.friction < 0
            ):
                errors.append(
                    f"Joint {joint.name!r} effort, velocity, damping and friction "
                    "must be non-negative"
                )

        by_joint_name = {joint.name: joint for joint in self.joints}
        for joint in self.joints:
            if not joint.mimic_joint:
                continue
            source = by_joint_name.get(joint.mimic_joint)
            if source is None:
                errors.append(
                    f"Joint {joint.name!r} mimics missing joint {joint.mimic_joint!r}"
                )
                continue
            if source is joint:
                errors.append(f"Joint {joint.name!r} cannot mimic itself")
            if joint.type not in {"revolute", "continuous", "prismatic"}:
                errors.append(f"Joint {joint.name!r} cannot be a scalar mimic joint")
            if source.type not in {"revolute", "continuous", "prismatic"}:
                errors.append(
                    f"Joint {joint.name!r} mimics non-scalar joint {source.name!r}"
                )
            if not math.isfinite(joint.mimic_multiplier) or not math.isfinite(
                joint.mimic_offset
            ):
                errors.append(
                    f"Joint {joint.name!r} contains non-finite mimic settings"
                )
            if joint.mimic_auto:
                source_limits = self._joint_preview_limits(source)
                if (
                    source_limits is None
                    or math.isclose(source_limits[0], source_limits[1])
                ):
                    errors.append(
                        f"Joint {joint.name!r} cannot auto-map a zero-range source joint"
                    )

        # Mimic dependencies are independent of the link tree and need their
        # own cycle check.
        for start in by_joint_name:
            seen: set[str] = set()
            current = start
            while current in by_joint_name and by_joint_name[current].mimic_joint:
                if current in seen:
                    errors.append(
                        f"Mimic joint dependency contains a cycle at {current!r}"
                    )
                    break
                seen.add(current)
                current = str(by_joint_name[current].mimic_joint)

        for joint in self.joints:
            if not joint.drive_source_joint:
                continue
            source = by_joint_name.get(joint.drive_source_joint)
            if source is None:
                errors.append(
                    f"Joint {joint.name!r} drives from missing joint "
                    f"{joint.drive_source_joint!r}"
                )
                continue
            if source is joint:
                errors.append(f"Joint {joint.name!r} cannot drive from itself")
            if joint.type != "continuous":
                errors.append(
                    f"Joint {joint.name!r} must be continuous for lever speed drive"
                )
            if joint.mimic_joint:
                errors.append(
                    f"Joint {joint.name!r} cannot use mimic and lever speed drive together"
                )
            if source.type not in {"revolute", "prismatic"}:
                errors.append(
                    f"Joint {joint.name!r} drive source {source.name!r} must be revolute or prismatic"
                )
            source_limits = self._joint_preview_limits(source)
            if source_limits is None or math.isclose(
                source_limits[0], source_limits[1]
            ):
                errors.append(
                    f"Joint {joint.name!r} drive source {source.name!r} requires a finite non-zero range"
                )
            if not math.isfinite(joint.drive_max_velocity) or joint.drive_max_velocity <= 0:
                errors.append(
                    f"Joint {joint.name!r} drive maximum velocity must be positive and finite"
                )
            if not math.isfinite(joint.drive_deadband) or not (
                0.0 <= joint.drive_deadband < 1.0
            ):
                errors.append(
                    f"Joint {joint.name!r} drive deadband must be in the range 0..<1"
                )

        roots = self.root_candidates()
        if self.links and len(roots) != 1:
            errors.append(f"Expected exactly one root link; found {len(roots)}")
        elif roots and self.root_link != roots[0]:
            errors.append(
                f"Configured root {self.root_link!r} differs from tree root {roots[0]!r}"
            )

        # Cycle/disconnection check independent of FK so all errors can be shown.
        if self.root_link in self.links:
            visited: set[str] = set()
            active: set[str] = set()

            def visit(link_name: str) -> None:
                if link_name in active:
                    errors.append(f"Kinematic tree contains a cycle at {link_name!r}")
                    return
                if link_name in visited:
                    return
                active.add(link_name)
                for child in adjacency.get(link_name, ()):  # pragma: no branch
                    visit(child)
                active.remove(link_name)
                visited.add(link_name)

            visit(self.root_link)
            disconnected = [name for name in self.links if name not in visited]
            if disconnected:
                errors.append(
                    f"Links are disconnected from root {self.root_link!r}: "
                    + ", ".join(disconnected)
                )

        owners: dict[str, str] = {}
        for link in self.links.values():
            for part_id in link.part_ids:
                if part_id not in self.parts:
                    errors.append(f"Link {link.name!r} references missing part {part_id!r}")
                    continue
                if part_id in owners:
                    errors.append(
                        f"Part {part_id!r} belongs to both {owners[part_id]!r} and {link.name!r}"
                    )
                owners[part_id] = link.name
                if self.parts[part_id].link_name != link.name:
                    errors.append(f"Part {part_id!r} has an inconsistent link assignment")
        for part in self.parts.values():
            if part.link_name is not None and part.link_name not in self.links:
                errors.append(
                    f"Part {part.id!r} references missing link {part.link_name!r}"
                )
            elif part.link_name is not None and owners.get(part.id) != part.link_name:
                errors.append(f"Part {part.id!r} is absent from its assigned LinkSpec")

        # Preserve order while avoiding repeated recursive diagnostics.
        errors = list(dict.fromkeys(errors))
        if errors and raise_on_error:
            raise ProjectValidationError(errors)
        return errors

    def assert_valid(self, *, check_names: bool = True) -> None:
        self.validate(raise_on_error=True, check_names=check_names)


__all__ = [
    "CollisionCandidate",
    "CollisionSweepFinding",
    "JointSpec",
    "LinkSpec",
    "ProjectValidationError",
    "RobotProject",
    "ScenePart",
    "VALID_JOINT_TYPES",
    "apply_transform",
    "axis_angle_matrix",
    "is_valid_name",
    "matrix_rpy",
    "rpy_matrix",
    "sanitize_name",
    "transform_matrix",
]
