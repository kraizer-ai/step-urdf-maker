import json
import struct
from pathlib import Path
import math
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from urdf_maker.model import JointSpec, LinkSpec, RobotProject, ScenePart
from urdf_maker.urdf_io import export_urdf, load_mesh, load_urdf, resolve_mesh_path


def write_binary_triangle(path: Path):
    header = b"test".ljust(80, b"\0")
    record = struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        100.0,
        0.0,
        0.0,
        0.0,
        100.0,
        0.0,
        0,
    )
    path.write_bytes(header + struct.pack("<I", 1) + record)


class MeshReaderTests(unittest.TestCase):
    def test_stl_obj_and_ascii_ply(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stl = root / "triangle.stl"
            write_binary_triangle(stl)
            vertices, triangles = load_mesh(stl)
            self.assertEqual(vertices.shape, (3, 3))
            np.testing.assert_array_equal(triangles, ((0, 1, 2),))

            obj = root / "quad.obj"
            obj.write_text(
                "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n",
                encoding="utf-8",
            )
            vertices, triangles = load_mesh(obj)
            self.assertEqual(vertices.shape, (4, 3))
            self.assertEqual(triangles.shape, (2, 3))

            ply = root / "triangle.ply"
            ply.write_text(
                "ply\nformat ascii 1.0\n"
                "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
                "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
                "0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
                encoding="ascii",
            )
            vertices, triangles = load_mesh(ply)
            np.testing.assert_allclose(vertices[2], (0.0, 1.0, 0.0))
            np.testing.assert_array_equal(triangles, ((0, 1, 2),))

    def test_minimal_collada_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            dae = Path(directory) / "triangle.dae"
            dae.write_text(
                """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema">
  <asset><unit meter="0.001"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id="g"><mesh>
    <source id="positions"><float_array id="a" count="9">0 0 0 100 0 0 0 100 0</float_array>
      <technique_common><accessor source="#a" count="3" stride="3"/></technique_common></source>
    <vertices id="v"><input semantic="POSITION" source="#positions"/></vertices>
    <triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles>
  </mesh></geometry></library_geometries>
</COLLADA>""",
                encoding="utf-8",
            )
            vertices, triangles = load_mesh(dae)
            np.testing.assert_allclose(vertices[1], (0.1, 0.0, 0.0))
            np.testing.assert_array_equal(triangles, ((0, 1, 2),))


class UrdfLoadTests(unittest.TestCase):
    def test_loads_primitives_relative_mesh_scale_origins_and_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_binary_triangle(root / "triangle.stl")
            urdf = root / "robot.urdf"
            urdf.write_text(
                """<robot name="test_robot">
  <material name="blue"><color rgba="0 0 1 0.5"/></material>
  <link name="base">
    <visual name="base_box"><origin xyz="0.1 0 0"/><geometry><box size="1 2 3"/></geometry><material name="blue"/></visual>
  </link>
  <link name="finger">
    <visual><origin xyz="0 0.2 0"/><geometry><mesh filename="triangle.stl" scale="0.001 0.001 0.001"/></geometry></visual>
  </link>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="finger"/><origin xyz="1 0 0"/><axis xyz="1 0 0"/>
    <limit lower="0" upper="0.5" effort="10" velocity="2"/>
  </joint>
</robot>""",
                encoding="utf-8",
            )
            project = load_urdf(urdf, strict=True)
            self.assertEqual(project.root_link, "base")
            self.assertEqual(len(project.parts), 2)
            self.assertEqual(project.joint("slide").upper, 0.5)
            box_part = project.parts[project.links["base"].part_ids[0]]
            np.testing.assert_allclose(box_part.color, (0.0, 0.0, 1.0, 0.5))
            np.testing.assert_allclose(box_part.bounds[0], (-0.4, -1.0, -1.5))
            finger_part = project.parts[project.links["finger"].part_ids[0]]
            np.testing.assert_allclose(finger_part.vertices_zero[0], (1.0, 0.2, 0.0))
            np.testing.assert_allclose(finger_part.vertices_zero[1], (1.1, 0.2, 0.0))
            moved = project.transformed_part_vertices(finger_part.id, {"slide": 0.3})
            np.testing.assert_allclose(moved[0], (1.3, 0.2, 0.0))

    def test_resolves_package_uri_from_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "my_robot"
            (package / "meshes").mkdir(parents=True)
            write_binary_triangle(package / "meshes" / "part.stl")
            urdf = root / "elsewhere" / "robot.urdf"
            urdf.parent.mkdir()
            urdf.write_text("<robot name='r'><link name='base'/></robot>", encoding="utf-8")
            resolved = resolve_mesh_path(
                "package://my_robot/meshes/part.stl", urdf, {"my_robot": package}
            )
            self.assertEqual(resolved, (package / "meshes" / "part.stl").resolve())

    def test_warns_when_round_trip_cannot_preserve_urdf_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf = Path(directory) / "extended.urdf"
            urdf.write_text(
                """<robot name="extended">
  <link name="base"><inertial><mass value="1"/></inertial>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
  <link name="tip"/>
  <joint name="fixed" type="fixed"><parent link="base"/><child link="tip"/>
  </joint>
  <transmission name="drive"/>
</robot>""",
                encoding="utf-8",
            )
            project = load_urdf(urdf)
            warnings = "\n".join(project.metadata["warnings"])
            self.assertIn("inertial", warnings)
            self.assertIn("collision", warnings)
            self.assertIn("transmission", warnings)

    def test_loads_mimic_joint_without_discard_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            urdf = Path(directory) / "mimic.urdf"
            urdf.write_text(
                """<robot name="mimic_robot">
  <link name="base"/><link name="handle"/><link name="axle"/>
  <joint name="handle_joint" type="revolute">
    <parent link="base"/><child link="handle"/><axis xyz="0 0 1"/>
    <limit lower="-3.141592653589793" upper="3.141592653589793" effort="1" velocity="1"/>
  </joint>
  <joint name="axle_joint" type="prismatic">
    <parent link="base"/><child link="axle"/><axis xyz="1 0 0"/>
    <limit lower="-0.05" upper="0.05" effort="1" velocity="1"/>
    <mimic joint="handle_joint" multiplier="0.015915494309" offset="0"/>
  </joint>
</robot>""",
                encoding="utf-8",
            )

            project = load_urdf(urdf, strict=True)

            mimic = project.joint("axle_joint")
            self.assertEqual(mimic.mimic_joint, "handle_joint")
            self.assertAlmostEqual(mimic.mimic_multiplier, 0.015915494309)
            self.assertFalse(mimic.mimic_auto)
            self.assertNotIn("mimic", "\n".join(project.metadata["warnings"]))


class UrdfExportTests(unittest.TestCase):
    def test_exports_package_binary_stl_inertial_and_round_trips_zero_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            local = np.array(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0)))
            base_part = ScenePart("base_part", "Base", local, ((0, 1, 2),), link_name="base link")
            finger_world = local + np.array((1.0, 0.0, 0.0))
            finger_part = ScenePart(
                "finger_part", "Finger", finger_world, ((0, 1, 2),), link_name="finger link"
            )
            project = RobotProject(
                "My Robot",
                parts=[base_part, finger_part],
                links=[
                    LinkSpec("base link", ["base_part"]),
                    LinkSpec("finger link", ["finger_part"]),
                ],
                joints=[
                    JointSpec(
                        "finger slide",
                        "prismatic",
                        "base link",
                        "finger link",
                        origin_xyz=(1.0, 0.0, 0.0),
                        axis=(1.0, 0.0, 0.0),
                        lower=-0.1,
                        upper=0.2,
                    )
                ],
                root_link="base link",
            )
            package_dir = Path(directory) / "output"
            output = export_urdf(
                project,
                package_dir,
                package_name="My Robot Description",
                include_inertial=True,
            )
            self.assertTrue(output.is_file())
            self.assertTrue((package_dir / "package.xml").is_file())
            tree = ET.parse(output)
            self.assertEqual(tree.getroot().get("name"), "My_Robot")
            self.assertEqual(
                {node.get("name") for node in tree.findall("link")},
                {"base_link", "finger_link"},
            )
            self.assertIsNotNone(tree.find("link/inertial"))
            mesh_uri = tree.find("link/visual/geometry/mesh").get("filename")
            self.assertTrue(mesh_uri.startswith("package://my_robot_description/meshes/"))
            stl_path = package_dir / "meshes" / "finger_link.stl"
            self.assertEqual(stl_path.stat().st_size, 84 + 50)

            loaded = load_urdf(
                output,
                package_dirs={"my_robot_description": package_dir},
                strict=True,
            )
            loaded_finger = loaded.parts[loaded.links["finger_link"].part_ids[0]]
            np.testing.assert_allclose(loaded_finger.vertices_zero, finger_world, atol=1e-7)
            self.assertEqual(loaded.joint("finger_slide").lower, -0.1)

    def test_exports_auto_mimic_as_standard_multiplier_and_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            project = RobotProject(
                "steering",
                links=[LinkSpec("base"), LinkSpec("handle"), LinkSpec("axle")],
                joints=[
                    JointSpec(
                        "handle_joint",
                        "revolute",
                        "base",
                        "handle",
                        lower=-math.pi,
                        upper=math.pi,
                    ),
                    JointSpec(
                        "axle_joint",
                        "prismatic",
                        "base",
                        "axle",
                        lower=-0.05,
                        upper=0.05,
                        mimic_joint="handle_joint",
                        mimic_auto=True,
                        mimic_reverse=True,
                    ),
                ],
                root_link="base",
            )
            output = export_urdf(project, Path(directory) / "output")
            mimic = ET.parse(output).find("joint[@name='axle_joint']/mimic")

            self.assertIsNotNone(mimic)
            self.assertEqual(mimic.get("joint"), "handle_joint")
            self.assertAlmostEqual(float(mimic.get("multiplier")), -0.05 / math.pi)
            self.assertAlmostEqual(float(mimic.get("offset")), 0.0)

    def test_exports_joint_dynamics_and_mechanism_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            project = RobotProject(
                "conveyor",
                links=[LinkSpec("base"), LinkSpec("roller")],
                joints=[
                    JointSpec(
                        "roller_joint",
                        "continuous",
                        "base",
                        "roller",
                        damping=0.02,
                        friction=0.03,
                    )
                ],
                root_link="base",
                metadata={
                    "mechanisms": [
                        {
                            "type": "conveyor",
                            "link": "roller",
                            "joint": "roller_joint",
                            "simulation_role": "conveyor_velocity",
                        }
                    ]
                },
            )
            package = Path(directory) / "output"
            output = export_urdf(project, package)
            dynamics = ET.parse(output).find("joint[@name='roller_joint']/dynamics")

            self.assertIsNotNone(dynamics)
            self.assertEqual(float(dynamics.get("damping")), 0.02)
            self.assertEqual(float(dynamics.get("friction")), 0.03)
            manifest = json.loads(
                (package / "config" / "mechanisms.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["format"], "step-urdf-maker-mechanisms")
            self.assertEqual(manifest["mechanisms"][0]["type"], "conveyor")

            loaded = load_urdf(output, strict=True)
            self.assertEqual(loaded.joint("roller_joint").damping, 0.02)
            self.assertEqual(loaded.joint("roller_joint").friction, 0.03)


if __name__ == "__main__":
    unittest.main()
