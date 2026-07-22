from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from urdf_maker.step_loader import StepPart, _transform_points, load_step


def test_step_part_vertices_zero_alias_and_validation() -> None:
    part = StepPart(
        id="p1",
        name="part",
        vertices=np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0))),
        triangles=np.array(((0, 1, 2),)),
        color=(1.2, -0.1, 0.5),
    )
    assert part.vertices_zero is part.vertices
    assert part.vertices.dtype == np.float64
    assert part.triangles.dtype == np.int64
    assert part.color == (1.0, 0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="outside vertices"):
        StepPart("bad", "bad", part.vertices, np.array(((0, 1, 3),)))


def test_small_affine_transform_matches_homogeneous_math() -> None:
    points = np.array(((1.0, 2.0, 3.0), (-2.0, 0.5, 4.0)))
    matrix = np.array(
        (
            (0.0, -1.0, 0.0, 10.0),
            (1.0, 0.0, 0.0, -5.0),
            (0.0, 0.0, 2.0, 0.25),
            (0.0, 0.0, 0.0, 1.0),
        )
    )

    actual = _transform_points(points, matrix)

    np.testing.assert_allclose(
        actual,
        ((8.0, -4.0, 6.25), (9.5, -7.0, 8.25)),
    )


def _write_two_occurrence_step(path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.gp import gp_Trsf, gp_Vec

    application = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("step-loader-test"))
    application.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    assembly = shape_tool.NewShape()
    TDataStd_Name.Set_s(assembly, TCollection_ExtendedString("fixture"))
    definition = shape_tool.AddShape(
        BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape(), False
    )
    TDataStd_Name.Set_s(definition, TCollection_ExtendedString("jaw_definition"))

    for x, name in ((0.0, "left_jaw"), (100.0, "right_jaw")):
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(x, 0.0, 0.0))
        occurrence = shape_tool.AddComponent(
            assembly, definition, TopLoc_Location(transform)
        )
        TDataStd_Name.Set_s(occurrence, TCollection_ExtendedString(name))
    shape_tool.UpdateAssemblies()

    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    assert writer.Transfer(document)
    assert writer.Write(str(path)) == IFSelect_RetDone


def _write_cylinder_step(path: Path) -> None:
    pytest.importorskip("OCP")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool
    from OCP.gp import gp_Trsf, gp_Vec

    application = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("cylinder-axis-test"))
    application.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    assembly = shape_tool.NewShape()
    definition = shape_tool.AddShape(
        BRepPrimAPI_MakeCylinder(35.0, 80.0).Shape(), False
    )
    TDataStd_Name.Set_s(definition, TCollection_ExtendedString("shaft_definition"))
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(10.0, 20.0, 30.0))
    occurrence = shape_tool.AddComponent(
        assembly, definition, TopLoc_Location(transform)
    )
    TDataStd_Name.Set_s(occurrence, TCollection_ExtendedString("shaft"))
    shape_tool.UpdateAssemblies()

    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    assert writer.Transfer(document)
    assert writer.Write(str(path)) == IFSelect_RetDone


def test_xcaf_restores_occurrence_names_and_world_placements(tmp_path: Path) -> None:
    source = tmp_path / "two_jaws.step"
    _write_two_occurrence_step(source)

    result = load_step(source, linear_deflection=0.001)

    assert result.units == "m"
    assert result.metadata["import_mode"] == "xcaf"
    assert result.metadata["occurrence_count"] == 2
    assert result.metadata["source_units"] == ["millimetre"]
    assert result.warnings == []
    assert [part.name for part in result.parts] == ["left_jaw", "right_jaw"]
    assert len({part.source_label for part in result.parts}) == 2
    np.testing.assert_allclose(result.parts[0].bounds[0], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(result.parts[0].bounds[1], (0.01, 0.02, 0.03))
    np.testing.assert_allclose(result.parts[1].bounds[0], (0.1, 0.0, 0.0))
    np.testing.assert_allclose(result.parts[1].bounds[1], (0.11, 0.02, 0.03))

    project = result.to_robot_project()
    assert len(project.parts) == 2
    assert project.source_kind == "step"
    np.testing.assert_allclose(
        project.parts[result.parts[1].id].vertices_zero, result.parts[1].vertices
    )


def test_step_loader_preserves_exact_cylindrical_feature_axis(tmp_path: Path) -> None:
    source = tmp_path / "shaft.step"
    _write_cylinder_step(source)

    result = load_step(source, linear_deflection=0.001)

    assert len(result.parts) == 1
    axes = result.parts[0].feature_axes
    assert axes
    cylindrical = min(axes, key=lambda item: abs(item["radius"] - 0.035))
    assert cylindrical["kind"] == "cylinder"
    assert cylindrical["radius"] == pytest.approx(0.035)
    assert cylindrical["length"] == pytest.approx(0.08)
    np.testing.assert_allclose(cylindrical["origin"], (0.01, 0.02, 0.07))
    np.testing.assert_allclose(cylindrical["direction"], (0.0, 0.0, 1.0))


def test_plain_stepcontrol_fallback_enumerates_solids(tmp_path: Path) -> None:
    source = tmp_path / "two_solids.step"
    _write_two_occurrence_step(source)

    result = load_step(source, linear_deflection=0.001, prefer_xcaf=False)

    assert result.metadata["import_mode"] == "stepcontrol-fallback"
    assert len(result.parts) == 2
    assert [part.name for part in result.parts] == ["solid_0001", "solid_0002"]
    assert result.parts[0].triangles.shape[1] == 3
    assert result.parts[1].vertices.max(axis=0)[0] == pytest.approx(0.11)


def test_project_wsr_step_has_assembly_occurrences_in_metres() -> None:
    source = (
        Path(__file__).parents[1]
        / "data"
        / "wsr-0002898 (simplified extended) 2025-11-23.STEP"
    )
    if not source.is_file():
        pytest.skip("project sample STEP is not available")

    result = load_step(source, linear_deflection=0.001)

    assert result.metadata["import_mode"] == "xcaf"
    assert result.metadata["root_count"] == 1
    assert result.metadata["occurrence_count"] == 58
    assert len(result.parts) == 58
    assert sum(len(part.triangles) for part in result.parts) > 200_000
    assert any(part.name == "bvme-0002591" for part in result.parts)
    lower, upper = np.asarray(result.metadata["bounds_m"])
    np.testing.assert_allclose(lower, (-0.5335, -0.8080, -0.5675), atol=2e-4)
    np.testing.assert_allclose(upper, (0.5335, 0.0835, 0.4015), atol=2e-4)
