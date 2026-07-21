import math
import unittest

import numpy as np

from urdf_maker.model import (
    JointSpec,
    LinkSpec,
    ProjectValidationError,
    RobotProject,
    ScenePart,
    apply_transform,
    rpy_matrix,
    sanitize_name,
)


def triangle_part(identifier="part", link_name=None, offset=(0.0, 0.0, 0.0)):
    vertices = np.array(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)))
    return ScenePart(
        identifier,
        identifier,
        vertices + np.asarray(offset),
        np.array(((0, 1, 2),)),
        link_name=link_name,
    )


class TransformTests(unittest.TestCase):
    def test_rpy_uses_urdf_fixed_axis_order(self):
        rotation = rpy_matrix((0.0, 0.0, math.pi / 2.0))
        np.testing.assert_allclose(rotation @ np.array((1.0, 0.0, 0.0)), (0.0, 1.0, 0.0), atol=1e-12)

    def test_apply_transform_accepts_empty_mesh(self):
        result = apply_transform(np.empty((0, 3)), np.eye(4))
        self.assertEqual(result.shape, (0, 3))

    def test_name_sanitizing(self):
        self.assertEqual(sanitize_name("left finger #1"), "left_finger_1")
        self.assertEqual(sanitize_name("123"), "item_123")


class ProjectKinematicsTests(unittest.TestCase):
    def test_prismatic_and_revolute_forward_kinematics(self):
        project = RobotProject(
            "robot",
            links=[LinkSpec("base"), LinkSpec("slide"), LinkSpec("tip")],
            joints=[
                JointSpec(
                    "slide_joint",
                    "prismatic",
                    "base",
                    "slide",
                    origin_xyz=(1.0, 0.0, 0.0),
                    axis=(0.0, 1.0, 0.0),
                    lower=-1.0,
                    upper=1.0,
                    position=0.25,
                ),
                JointSpec(
                    "tip_joint",
                    "revolute",
                    "slide",
                    "tip",
                    origin_xyz=(0.0, 0.0, 1.0),
                    axis=(0.0, 0.0, 1.0),
                    lower=-math.pi,
                    upper=math.pi,
                    position=math.pi / 2.0,
                ),
            ],
            root_link="base",
        )
        fk = project.forward_kinematics()
        np.testing.assert_allclose(fk["slide"][:3, 3], (1.0, 0.25, 0.0))
        np.testing.assert_allclose(fk["tip"][:3, 3], (1.0, 0.25, 1.0))
        np.testing.assert_allclose(
            fk["tip"][:3, :3] @ np.array((1.0, 0.0, 0.0)),
            (0.0, 1.0, 0.0),
            atol=1e-12,
        )

    def test_world_zero_part_moves_around_link_origin(self):
        part = triangle_part(link_name="finger", offset=(1.0, 0.0, 0.0))
        project = RobotProject(
            "robot",
            parts=[part],
            links=[LinkSpec("base"), LinkSpec("finger", [part.id])],
            joints=[
                JointSpec(
                    "finger_joint",
                    "revolute",
                    "base",
                    "finger",
                    origin_xyz=(1.0, 0.0, 0.0),
                    axis=(0.0, 0.0, 1.0),
                    lower=-math.pi,
                    upper=math.pi,
                )
            ],
            root_link="base",
        )
        moved = project.transformed_part_vertices(
            part.id, {"finger_joint": math.pi / 2.0}
        )
        np.testing.assert_allclose(moved[0], (1.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(moved[1], (1.0, 0.1, 0.0), atol=1e-12)
        local, triangles = project.link_vertices_local("finger")
        np.testing.assert_allclose(local[0], (0.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_array_equal(triangles, ((0, 1, 2),))

    def test_position_nudge_clamps_to_limits(self):
        project = RobotProject(
            "robot",
            links=[LinkSpec("base"), LinkSpec("finger")],
            joints=[
                JointSpec(
                    "j", "prismatic", "base", "finger", lower=-0.2, upper=0.3
                )
            ],
            root_link="base",
        )
        self.assertEqual(project.nudge_joint("j", 1.0), 0.3)
        self.assertEqual(project.nudge_joint("j", -2.0), -0.2)


class ProjectEditingTests(unittest.TestCase):
    def test_create_link_with_unknown_part_is_atomic(self):
        project = RobotProject("robot", parts=[triangle_part("known")])

        with self.assertRaises(KeyError):
            project.create_link("new_link", ["missing"])

        self.assertNotIn("new_link", project.links)
        self.assertIsNone(project.root_link)
        self.assertIsNone(project.parts["known"].link_name)

    def test_assignment_creation_and_merge(self):
        first = triangle_part("first")
        second = triangle_part("second")
        project = RobotProject("robot", parts=[first, second])
        project.create_link("base", ["first"])
        project.create_link("moving", ["second"])
        project.joints.append(JointSpec("fixed", "fixed", "base", "moving"))
        self.assertEqual(first.link_name, "base")
        self.assertEqual(second.link_name, "moving")
        project.assign_parts(["first"], "moving")
        self.assertEqual(project.links["base"].part_ids, [])
        self.assertEqual(project.links["moving"].part_ids, ["second", "first"])
        project.merge_links("base", "moving")
        self.assertNotIn("moving", project.links)
        self.assertEqual(project.joints, [])
        self.assertEqual(set(project.links["base"].part_ids), {"first", "second"})
        self.assertEqual(project.validate(), [])

    def test_merge_preserves_crossing_joint_zero_pose(self):
        project = RobotProject(
            "robot",
            links=[LinkSpec("base"), LinkSpec("spacer"), LinkSpec("tip")],
            joints=[
                JointSpec(
                    "spacer_fixed",
                    "fixed",
                    "base",
                    "spacer",
                    origin_xyz=(1.0, 0.0, 0.0),
                ),
                JointSpec(
                    "tip_slide",
                    "prismatic",
                    "spacer",
                    "tip",
                    origin_xyz=(2.0, 0.0, 0.0),
                    lower=-1.0,
                    upper=1.0,
                ),
            ],
            root_link="base",
        )
        before = project.forward_kinematics(zero=True)["tip"]
        project.merge_links("base", "spacer")
        self.assertEqual(len(project.joints), 1)
        self.assertEqual(project.joints[0].parent, "base")
        np.testing.assert_allclose(project.joints[0].origin_xyz, (3.0, 0.0, 0.0))
        np.testing.assert_allclose(project.forward_kinematics(zero=True)["tip"], before)

    def test_validation_reports_tree_limit_and_name_errors(self):
        project = RobotProject(
            "bad robot",
            links=[LinkSpec("base link"), LinkSpec("child")],
            joints=[
                JointSpec(
                    "bad joint",
                    "prismatic",
                    "base link",
                    "child",
                    axis=(0.0, 0.0, 0.0),
                    lower=1.0,
                    upper=-1.0,
                )
            ],
            root_link="base link",
        )
        errors = "\n".join(project.validate())
        self.assertIn("Robot name", errors)
        self.assertIn("Link name", errors)
        self.assertIn("zero axis", errors)
        self.assertIn("lower limit", errors)
        with self.assertRaises(ProjectValidationError):
            project.assert_valid()

    def test_validation_detects_disconnected_tree(self):
        project = RobotProject(
            "robot",
            links=[LinkSpec("root"), LinkSpec("orphan")],
            root_link="root",
        )
        errors = "\n".join(project.validate())
        self.assertIn("exactly one root", errors)
        self.assertIn("disconnected", errors)

    def test_validation_does_not_silently_accept_duplicate_links_or_parts(self):
        one = triangle_part("same")
        two = triangle_part("same")
        project = RobotProject(
            "robot",
            parts=[one, two],
            links=[LinkSpec("base"), LinkSpec("base")],
            root_link="base",
        )
        errors = "\n".join(project.validate())
        self.assertIn("Duplicate part id", errors)
        self.assertIn("Duplicate link name", errors)


if __name__ == "__main__":
    unittest.main()
