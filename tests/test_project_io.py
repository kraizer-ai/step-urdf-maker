from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from urdf_maker.model import JointSpec, LinkSpec, RobotProject, ScenePart
from urdf_maker.project_io import apply_project_config, load_project_config, save_project


def _project(source: Path) -> RobotProject:
    part = ScenePart(
        "part",
        "part",
        np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float),
        np.asarray(((0, 1, 2),), dtype=np.int64),
        link_name="base_link",
    )
    return RobotProject(
        "robot",
        parts=[part],
        links=[LinkSpec("base_link", ["part"]), LinkSpec("moving_link")],
        joints=[
            JointSpec(
                "moving_joint",
                "prismatic",
                "base_link",
                "moving_link",
                axis=(1, 0, 0),
                lower=-0.1,
                upper=0.1,
                position=0.02,
            )
        ],
        root_link="base_link",
        source_path=str(source),
        source_kind="step",
    )


def test_project_round_trip_config(tmp_path: Path) -> None:
    source = tmp_path / "source.step"
    source.write_text("dummy", encoding="utf-8")
    original = _project(source)
    path = save_project(original, tmp_path / "robot.urdfmaker.json")
    payload, resolved_source, kind = load_project_config(path)
    assert resolved_source == source.resolve()
    assert kind == "step"

    fresh = _project(source)
    fresh.joints.clear()
    warnings = apply_project_config(fresh, payload)
    assert warnings == []
    assert len(fresh.joints) == 1
    assert fresh.joints[0].position == 0.02
    assert fresh.validate(check_names=False) == []


def test_project_file_is_human_readable(tmp_path: Path) -> None:
    source = tmp_path / "source.urdf"
    source.write_text("<robot name='r'/>", encoding="utf-8")
    path = save_project(_project(source), tmp_path / "robot.urdfmaker.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "step-urdf-maker"
    assert payload["source"]["path"] == "source.urdf"
    assert payload["parts"]["part"]["signature"]["triangle_count"] == 1


def test_changed_part_signature_is_kept_safely_on_root(tmp_path: Path) -> None:
    source = tmp_path / "source.step"
    source.write_text("dummy", encoding="utf-8")
    original = _project(source)
    original.assign_parts(["part"], "moving_link")
    path = save_project(original, tmp_path / "robot.urdfmaker.json")
    payload, _, _ = load_project_config(path)

    fresh = _project(source)
    fresh.parts["part"].name = "different occurrence"
    warnings = apply_project_config(fresh, payload)

    assert fresh.parts["part"].link_name == "base_link"
    assert "part" in fresh.links["base_link"].part_ids
    assert "part" not in fresh.links["moving_link"].part_ids
    assert any("저장 시점과 달라" in warning for warning in warnings)


def test_explicitly_unassigned_part_survives_project_round_trip(tmp_path: Path) -> None:
    source = tmp_path / "source.step"
    source.write_text("dummy", encoding="utf-8")
    original = _project(source)
    original.assign_parts(["part"], None)
    path = save_project(original, tmp_path / "manual.urdfmaker.json")
    payload, _, _ = load_project_config(path)

    fresh = _project(source)
    warnings = apply_project_config(fresh, payload)

    assert warnings == []
    assert fresh.parts["part"].link_name is None
    assert "part" not in fresh.links["base_link"].part_ids
