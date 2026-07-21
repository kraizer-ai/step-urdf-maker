from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import JointSpec, LinkSpec, RobotProject


PROJECT_VERSION = 1


def _portable_source_path(source_path: str | None, project_path: Path) -> str | None:
    if not source_path:
        return None
    source = Path(source_path).expanduser().resolve()
    try:
        return source.relative_to(project_path.parent.resolve()).as_posix()
    except ValueError:
        return str(source)


def _part_signature(part: Any) -> dict[str, Any]:
    bounds = part.bounds
    bounds_payload = None
    if bounds is not None:
        bounds_payload = [
            np.asarray(bounds[0], dtype=float).round(9).tolist(),
            np.asarray(bounds[1], dtype=float).round(9).tolist(),
        ]
    return {
        "name": part.name,
        "bounds_m": bounds_payload,
        "vertex_count": int(len(part.vertices_zero)),
        "triangle_count": int(len(part.triangles)),
    }


def _saved_part_matches(part: Any, state: Any) -> bool:
    if not isinstance(state, dict) or not isinstance(state.get("signature"), dict):
        # Version-1 files created before signatures were added remain readable.
        return True
    signature = state["signature"]
    if "name" in signature and str(signature["name"]) != part.name:
        return False
    if "vertex_count" in signature and int(signature["vertex_count"]) != len(part.vertices_zero):
        return False
    if "triangle_count" in signature and int(signature["triangle_count"]) != len(part.triangles):
        return False
    saved_bounds = signature.get("bounds_m")
    current_bounds = part.bounds
    if saved_bounds is None:
        return current_bounds is None
    if current_bounds is None:
        return False
    try:
        return bool(
            np.allclose(
                np.asarray(saved_bounds, dtype=float),
                np.asarray(current_bounds, dtype=float),
                rtol=1e-7,
                atol=1e-8,
            )
        )
    except (TypeError, ValueError):
        return False


def project_to_dict(project: RobotProject, project_path: str | Path) -> dict[str, Any]:
    destination = Path(project_path).resolve()
    return {
        "format": "step-urdf-maker",
        "version": PROJECT_VERSION,
        "name": project.name,
        "source": {
            "path": _portable_source_path(project.source_path, destination),
            "kind": project.source_kind,
        },
        "root_link": project.root_link,
        "links": [
            {"name": link.name, "part_ids": list(link.part_ids)}
            for link in project.links.values()
        ],
        "joints": [
            {
                "name": joint.name,
                "type": joint.type,
                "parent": joint.parent,
                "child": joint.child,
                "origin_xyz": np.asarray(joint.origin_xyz, dtype=float).tolist(),
                "origin_rpy": np.asarray(joint.origin_rpy, dtype=float).tolist(),
                "axis": np.asarray(joint.axis, dtype=float).tolist(),
                "lower": joint.lower,
                "upper": joint.upper,
                "effort": joint.effort,
                "velocity": joint.velocity,
                "position": joint.position,
            }
            for joint in project.joints
        ],
        "parts": {
            part.id: {
                "link": part.link_name,
                "visible": part.visible,
                "signature": _part_signature(part),
            }
            for part in project.parts.values()
        },
        "metadata": {
            key: value
            for key, value in project.metadata.items()
            if isinstance(value, (str, int, float, bool, list, dict, type(None)))
        },
    }


def save_project(project: RobotProject, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = project_to_dict(project, destination)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination


def load_project_config(path: str | Path) -> tuple[dict[str, Any], Path, str | None]:
    project_path = Path(path).expanduser().resolve()
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    if payload.get("format") != "step-urdf-maker":
        raise ValueError("STEP URDF Maker 프로젝트 파일이 아닙니다.")
    version = int(payload.get("version", 0))
    if version > PROJECT_VERSION:
        raise ValueError(
            f"프로젝트 버전 {version}은 이 프로그램이 지원하는 {PROJECT_VERSION}보다 새 버전입니다."
        )
    source = payload.get("source") or {}
    raw_source = source.get("path")
    source_path: Path | None = None
    if raw_source:
        candidate = Path(raw_source).expanduser()
        source_path = candidate if candidate.is_absolute() else project_path.parent / candidate
        source_path = source_path.resolve()
    return payload, source_path, source.get("kind")


def apply_project_config(
    project: RobotProject,
    payload: dict[str, Any],
) -> list[str]:
    """Apply saved topology and editor state to freshly reloaded source geometry."""

    warnings: list[str] = []
    project.name = str(payload.get("name") or project.name)
    part_state = payload.get("parts") or {}
    if not isinstance(part_state, dict):
        warnings.append("저장된 형상 상태가 올바르지 않아 표시 설정을 건너뜁니다.")
        part_state = {}
    incompatible_parts: set[str] = set()
    if isinstance(part_state, dict):
        for raw_part_id, state in part_state.items():
            part_id = str(raw_part_id)
            part = project.parts.get(part_id)
            if part is not None and not _saved_part_matches(part, state):
                incompatible_parts.add(part_id)
                warnings.append(
                    f"{part_id}: 원본 형상이 저장 시점과 달라 링크 배정을 복원하지 않았습니다."
                )
    saved_links = payload.get("links") or []
    links: dict[str, LinkSpec] = {}
    for item in saved_links:
        name = str(item.get("name") or "").strip()
        if not name or name in links:
            warnings.append(f"저장된 중복/빈 링크 이름을 건너뜀: {name!r}")
            continue
        part_ids = [str(part_id) for part_id in item.get("part_ids") or []]
        missing = [part_id for part_id in part_ids if part_id not in project.parts]
        if missing:
            warnings.append(
                f"{name}: 원본에서 찾지 못한 형상 {len(missing)}개를 제외했습니다."
            )
        links[name] = LinkSpec(
            name,
            [
                part_id
                for part_id in part_ids
                if part_id in project.parts and part_id not in incompatible_parts
            ],
        )
    if links:
        project.links = links
        for part in project.parts.values():
            part.link_name = None
        for link in project.links.values():
            for part_id in link.part_ids:
                project.parts[part_id].link_name = link.name

        saved_root = payload.get("root_link") or project.root_link
        if saved_root in project.links:
            unassigned = [part.id for part in project.parts.values() if part.link_name is None]
            explicitly_unassigned = {
                str(part_id)
                for part_id, state in part_state.items()
                if (
                    str(part_id) in project.parts
                    and str(part_id) not in incompatible_parts
                    and isinstance(state, dict)
                    and "link" in state
                    and state.get("link") is None
                )
            }
            fallback_parts = [
                part_id for part_id in unassigned if part_id not in explicitly_unassigned
            ]
            if fallback_parts:
                project.assign_parts(fallback_parts, saved_root)
                warnings.append(
                    f"새 형상 또는 변경된 형상 {len(fallback_parts)}개를 {saved_root}에 유지했습니다."
                )

    joints: list[JointSpec] = []
    for item in payload.get("joints") or []:
        try:
            joints.append(
                JointSpec(
                    name=item["name"],
                    type=item["type"],
                    parent=item["parent"],
                    child=item["child"],
                    origin_xyz=item.get("origin_xyz", (0, 0, 0)),
                    origin_rpy=item.get("origin_rpy", (0, 0, 0)),
                    axis=item.get("axis", (1, 0, 0)),
                    lower=item.get("lower"),
                    upper=item.get("upper"),
                    effort=item.get("effort", 100.0),
                    velocity=item.get("velocity", 1.0),
                    position=item.get("position", 0.0),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            warnings.append(f"관절 설정 하나를 복원하지 못했습니다: {exc}")
    if payload.get("joints") is not None:
        project.joints = joints
    project.root_link = payload.get("root_link") or project.root_link

    for part_id, state in part_state.items():
        part = project.parts.get(str(part_id))
        if part is not None and str(part_id) not in incompatible_parts:
            part.visible = bool((state or {}).get("visible", True))

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        project.metadata.update(metadata)
    return warnings


__all__ = [
    "PROJECT_VERSION",
    "apply_project_config",
    "load_project_config",
    "project_to_dict",
    "save_project",
]
