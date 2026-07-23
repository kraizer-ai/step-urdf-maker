from __future__ import annotations

import numpy as np

from urdf_maker.collision import MeshCollisionChecker
from urdf_maker.model import JointSpec, LinkSpec, RobotProject, ScenePart


_TETRA_TRIANGLES = np.asarray(
    ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
    dtype=np.int64,
)


def _tetra(identifier: str, link_name: str, vertices: list[tuple[float, ...]]):
    return ScenePart(
        identifier,
        identifier,
        np.asarray(vertices, dtype=float),
        _TETRA_TRIANGLES,
        link_name=link_name,
    )


def _cube(
    identifier: str,
    link_name: str,
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
) -> ScenePart:
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    vertices = np.asarray(
        (
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ),
        dtype=float,
    )
    triangles = np.asarray(
        (
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ),
        dtype=np.int64,
    )
    return ScenePart(identifier, identifier, vertices, triangles, link_name=link_name)


def test_mesh_narrow_phase_rejects_overlapping_bounds_without_contact() -> None:
    lower_corner = _tetra(
        "lower",
        "lower_link",
        [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    )
    upper_corner = _tetra(
        "upper",
        "upper_link",
        [(1, 1, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)],
    )
    project = RobotProject(
        "false_positive",
        parts=[lower_corner, upper_corner],
        links=[
            LinkSpec("base"),
            LinkSpec("lower_link", [lower_corner.id]),
            LinkSpec("upper_link", [upper_corner.id]),
        ],
        joints=[
            JointSpec("lower_mount", "fixed", "base", "lower_link"),
            JointSpec("upper_mount", "fixed", "base", "upper_link"),
        ],
        root_link="base",
    )

    assert len(project.self_collision_candidates()) == 1
    assert MeshCollisionChecker(project).collisions_at() == []


def test_mesh_sweep_reports_real_triangle_intersection() -> None:
    moving = _cube("moving", "moving_link", (0, 0, 0), (0.2, 0.2, 0.2))
    obstacle = _cube(
        "obstacle",
        "obstacle_link",
        (0.9, 0, 0),
        (1.1, 0.2, 0.2),
    )
    project = RobotProject(
        "mesh_sweep",
        parts=[moving, obstacle],
        links=[
            LinkSpec("base"),
            LinkSpec("moving_link", [moving.id]),
            LinkSpec("obstacle_link", [obstacle.id]),
        ],
        joints=[
            JointSpec(
                "slide",
                "prismatic",
                "base",
                "moving_link",
                axis=(1, 0, 0),
                lower=0.0,
                upper=1.0,
            ),
            JointSpec("obstacle_mount", "fixed", "base", "obstacle_link"),
        ],
        root_link="base",
    )
    checker = MeshCollisionChecker(project)

    current, motion, omitted = checker.sampled_self_collisions()

    assert current == []
    assert omitted == 0
    assert any(
        finding.joint_name == "slide"
        and {finding.candidate.link_a, finding.candidate.link_b}
        == {"moving_link", "obstacle_link"}
        for finding in motion
    )


def test_mesh_narrow_phase_does_not_report_face_contact_as_penetration() -> None:
    first = _cube("first", "first_link", (0, 0, 0), (1, 1, 1))
    touching = _cube("touching", "touching_link", (1, 0, 0), (2, 1, 1))
    project = RobotProject(
        "touching",
        parts=[first, touching],
        links=[
            LinkSpec("base"),
            LinkSpec("first_link", [first.id]),
            LinkSpec("touching_link", [touching.id]),
        ],
        joints=[
            JointSpec("first_mount", "fixed", "base", "first_link"),
            JointSpec("touching_mount", "fixed", "base", "touching_link"),
        ],
        root_link="base",
    )

    assert MeshCollisionChecker(project).collisions_at() == []


def test_mesh_sweep_checks_combined_joint_extremes() -> None:
    left = _cube("left", "left_link", (-1.2, 0, 0), (-1.0, 0.2, 0.2))
    right = _cube("right", "right_link", (1.0, 0, 0), (1.2, 0.2, 0.2))
    project = RobotProject(
        "combined_sweep",
        parts=[left, right],
        links=[
            LinkSpec("base"),
            LinkSpec("left_link", [left.id]),
            LinkSpec("right_link", [right.id]),
        ],
        joints=[
            JointSpec(
                "left_slide",
                "prismatic",
                "base",
                "left_link",
                axis=(1, 0, 0),
                lower=0.0,
                upper=1.05,
            ),
            JointSpec(
                "right_slide",
                "prismatic",
                "base",
                "right_link",
                axis=(-1, 0, 0),
                lower=0.0,
                upper=1.05,
            ),
        ],
        root_link="base",
    )

    _current, motion, _omitted = MeshCollisionChecker(
        project
    ).sampled_self_collisions()

    assert any(finding.joint_name is None for finding in motion)
