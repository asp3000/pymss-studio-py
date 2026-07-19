"""Workflow graph model helpers.

These functions manipulate the **graph** representation that
``worker_graph_workflows.py`` actually consumes:

    {
      "kind": "pymss-studio-graph",
      "version": 2,
      "defaults": {"inference_params": {...}},
      "graph": {
        "nodes": [ {id, type, position:{x,y}, data:{...}} ],
        "edges": [ {source:{nodeId,portId}, target:{nodeId,portId}} ],
        "viewport": {x,y,k}
      }
    }

Node types (mirrors worker_graph_workflows.EXECUTABLE_NODE_TYPES / UTILITY_NODE_TYPES):
  input_audio          (exactly one, id must be "input")   out: audio
  separate             data: {model, stems:[...], overlapSize}  in: input  out: stem:<stem> per stem
  audio_ensemble       data: {inputCount, weights:[], ensembleType}  in: input:0..N  out: audio
  audio_invert_phase  data: {}                               in: input   out: audio
  audio_normalize      data: {}                              in: input   out: audio
  load_audio_batch     data: {folder, recursive, sortFiles}  in: -        out: audio
  save_outputs         data: {outputs:{sourceRef: outputDir}}  in: save:<label> per entry
"""
from __future__ import annotations

import copy
import uuid
from typing import Any

KIND = "pymss-studio-graph"
VERSION = 2

UTILITY_KINDS = {
    "audio_ensemble",
    "audio_invert_phase",
    "audio_normalize",
    "load_audio_batch",
}

NODE_TITLES = {
    "input_audio": "原始输入",
    "separate": "分离",
    "audio_ensemble": "音频集成",
    "audio_invert_phase": "反相",
    "audio_normalize": "归一化",
    "load_audio_batch": "批量输入",
    "save_outputs": "保存输出",
    "note": "便签",
}


# --------------------------------------------------------------------------
# construction / normalization
# --------------------------------------------------------------------------


def _node(id_: str, type_: str, x: float, y: float, data: dict | None = None) -> dict[str, Any]:
    return {"id": id_, "type": type_, "position": {"x": x, "y": y},
            "data": data or {}}


def normalize_definition(value: Any) -> dict[str, Any]:
    """Coerce arbitrary input into a valid graph definition.

    Keeps existing input/save nodes if present (preserving ids/positions),
    otherwise creates them. Ensures every node has id/type/position/data and
    every edge has source/target objects.
    """
    if not isinstance(value, dict):
        value = {}
    out: dict[str, Any] = {
        "kind": KIND,
        "version": VERSION,
        "defaults": value.get("defaults") if isinstance(value.get("defaults"), dict) else {"inference_params": {}},
        "graph": {},
    }
    graph = value.get("graph") if isinstance(value.get("graph"), dict) else {}
    raw_nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    raw_edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []

    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for n in raw_nodes:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id") or "").strip()
        ntype = str(n.get("type") or "").strip()
        if not nid or not ntype or nid in seen:
            continue
        seen.add(nid)
        pos = n.get("position") if isinstance(n.get("position"), dict) else {}
        nodes.append({
            "id": nid,
            "type": ntype,
            "position": {"x": float(pos.get("x", 0) or 0), "y": float(pos.get("y", 0) or 0)},
            "data": n.get("data") if isinstance(n.get("data"), dict) else {},
        })

    # ensure input + save nodes exist
    ids = {n["id"] for n in nodes}
    if "input" not in ids:
        nodes.append(_node("input", "input_audio", 80, 160))
    else:
        # normalize the input node data
        for n in nodes:
            if n["id"] == "input":
                n["type"] = "input_audio"
                n["data"] = n.get("data") or {}
    if "save" not in ids:
        nodes.append(_node("save", "save_outputs", 980, 160, {"outputs": {}}))
    else:
        for n in nodes:
            if n["id"] == "save":
                n["type"] = "save_outputs"
                d = n.get("data") or {}
                if not isinstance(d.get("outputs"), dict):
                    d["outputs"] = {}
                n["data"] = d

    edges: list[dict[str, Any]] = []
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        s = e.get("source") if isinstance(e.get("source"), dict) else {}
        t = e.get("target") if isinstance(e.get("target"), dict) else {}
        sid = str(s.get("nodeId") or "").strip()
        tid = str(t.get("nodeId") or "").strip()
        if not sid or not tid:
            continue
        edges.append({
            "source": {"nodeId": sid, "portId": str(s.get("portId") or "").strip()},
            "target": {"nodeId": tid, "portId": str(t.get("portId") or "").strip()},
        })

    vp = graph.get("viewport") if isinstance(graph.get("viewport"), dict) else {}
    out["graph"] = {
        "nodes": nodes,
        "edges": edges,
        "viewport": {"x": float(vp.get("x", 0) or 0), "y": float(vp.get("y", 0) or 0),
                     "k": float(vp.get("k", 1) or 1)},
    }
    return out


def nodes(defn: dict[str, Any]) -> list[dict[str, Any]]:
    return defn.get("graph", {}).get("nodes", [])


def edges(defn: dict[str, Any]) -> list[dict[str, Any]]:
    return defn.get("graph", {}).get("edges", [])


def get_node(defn: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for n in nodes(defn):
        if n["id"] == node_id:
            return n
    return None


# --------------------------------------------------------------------------
# stems / instruments
# --------------------------------------------------------------------------
def parse_instruments(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        items = [str(x).strip() for x in value]
    else:
        text = str(value).strip()
        if not text:
            return []
        if "|" in text:
            items = [p.strip() for p in text.split("|")]
        elif "," in text:
            items = [p.strip() for p in text.split(",")]
        else:
            items = [text]
    return [it for it in items if it]


# --------------------------------------------------------------------------
# ports
# --------------------------------------------------------------------------


def source_ref_for_edge(defn: dict[str, Any], edge: dict[str, Any]) -> str:
    """Mirror worker_graph_workflows._source_ref_for_edge."""
    src = edge.get("source") if isinstance(edge.get("source"), dict) else {}
    node = get_node(defn, str(src.get("nodeId") or ""))
    if not node:
        return ""
    port = str(src.get("portId") or "").strip()
    ntype = node.get("type")
    if ntype == "input_audio":
        return "input"
    if ntype == "load_audio_batch":
        return f"utility:{node['id']}"
    if ntype == "separate" and port.startswith("stem:"):
        return f"{node['id']}.{port[5:]}"
    if ntype in UTILITY_KINDS and port == "audio":
        return f"utility:{node['id']}"
    return ""


def _safe_label(value: str) -> str:
    import re
    safe = re.sub(r"[\\/:\0<>\"|?*]+", "_", str(value or "")).strip()
    return safe or "stem"


# --------------------------------------------------------------------------
# available outputs (for input-source dropdowns + save mapping)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# validation (mirrors worker _validate_graph_definition rules)
# --------------------------------------------------------------------------
def validate_definition(defn: dict[str, Any]) -> list[str]:
    errs: list[str] = []
    ns = {n["id"]: n for n in nodes(defn)}
    if "input" not in ns:
        errs.append("工作流缺少输入节点。")
    if not any(n.get("type") == "save_outputs" for n in nodes(defn)):
        errs.append("工作流缺少保存输出节点。")

    incoming: dict[tuple[str, str], int] = {}
    valid_save_targets = 0
    for e in edges(defn):
        s = e.get("source") if isinstance(e.get("source"), dict) else {}
        t = e.get("target") if isinstance(e.get("target"), dict) else {}
        sid, sport = str(s.get("nodeId") or ""), str(s.get("portId") or "").strip()
        tid, tport = str(t.get("nodeId") or ""), str(t.get("portId") or "").strip()
        sn, tn = ns.get(sid), ns.get(tid)
        if not sn or not tn:
            errs.append(f"存在悬空连接: {sid}:{sport} -> {tid}:{tport}。")
            continue
        if not _source_port_valid(sn, sport):
            errs.append(f"源端口不可用: {sid}:{sport}。")
        if not _target_port_valid(tn, tport):
            errs.append(f"目标端口不可用: {tid}:{tport}。")
        key = (tid, tport)
        incoming[key] = incoming.get(key, 0) + 1
        if tn.get("type") == "save_outputs":
            valid_save_targets += 1

    dups = [f"{n}:{p}" for (n, p), c in incoming.items() if c > 1]
    if dups:
        errs.append(f"存在重复输入连接: {', '.join(sorted(dups))}。")

    missing: list[str] = []
    for nid, n in ns.items():
        for pid in _required_input_ports(n):
            if (nid, pid) not in incoming:
                missing.append(f"{nid}:{pid}")
    if missing:
        errs.append(f"工具节点存在未连接输入: {', '.join(sorted(missing))}。")

    if valid_save_targets <= 0:
        errs.append("工作流没有可保存的输出。")
    return errs


def _source_port_valid(node: dict[str, Any], port: str) -> bool:
    ntype = node.get("type")
    if ntype == "input_audio":
        return port == "audio"
    if ntype == "separate":
        if not port.startswith("stem:"):
            return False
        stem = port[5:].strip().lower()
        return bool(stem) and stem in {s.lower() for s in (node.get("data", {}).get("stems") or [])}
    if ntype in UTILITY_KINDS:
        return port == "audio"
    return False


def _target_port_valid(node: dict[str, Any], port: str) -> bool:
    ntype = node.get("type")
    if ntype == "separate":
        return port == "input"
    if ntype == "save_outputs":
        return port.startswith("save:")
    if ntype == "audio_ensemble":
        if not port.startswith("input:"):
            return False
        try:
            i = int(port.split(":", 1)[1])
        except (IndexError, ValueError):
            return False
        return 0 <= i < max(2, min(10, int((node.get("data", {}).get("inputCount") or 2))))
    if ntype in ("audio_invert_phase", "audio_normalize"):
        return port == "input"
    return False


def _required_input_ports(node: dict[str, Any]) -> list[str]:
    ntype = node.get("type")
    if ntype == "audio_ensemble":
        return [f"input:{i}" for i in range(max(2, min(10, int((node.get("data", {}).get("inputCount") or 2)))))]
    if ntype in ("audio_invert_phase", "audio_normalize"):
        return ["input"]
    return []


# --------------------------------------------------------------------------
# run payload
# --------------------------------------------------------------------------
def build_run_payload(
    *,
    defn: dict[str, Any],
    job_id: str,
    workflow_name: str,
    inputs: list[str],
    output_dir: str,
    output_format: str = "wav",
    output_layout: str = "folders",
    device: str = "auto",
    device_ids: list[int] | None = None,
    source: str = "modelscope",
    model_dir: str = "",
    audio_params: dict | None = None,
    use_tta: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    device_ids = device_ids or [0]
    tasks = [{"taskId": f"{job_id}_{i}", "input": inp} for i, inp in enumerate(inputs)]
    return {
        "taskId": job_id,
        "workflowName": workflow_name,
        "workflow": defn,
        "output": output_dir,
        "outputFormat": output_format,
        "outputLayout": output_layout,
        "modelDir": model_dir or None,
        "download": True,
        "source": source,
        "endpoint": None,
        "device": device,
        "deviceIds": device_ids,
        "useTta": use_tta,
        "debug": debug,
        "audioParams": audio_params or {},
        "tasks": tasks,
    }


# --------------------------------------------------------------------------
# save-outputs mapping helpers (used by editor + save node data)
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# used models / metrics (for the overview stage)
# --------------------------------------------------------------------------
def used_models(defn: dict[str, Any]) -> list[str]:
    """Ordered, de-duplicated list of model names referenced by separate nodes."""
    out: list[str] = []
    for n in nodes(defn):
        if n.get("type") == "separate":
            m = (n.get("data") or {}).get("model")
            if m and m not in out:
                out.append(m)
    return out


def graph_metrics(defn: dict[str, Any]) -> dict[str, int]:
    ns = nodes(defn)
    sep = [n for n in ns if n.get("type") == "separate"]
    util = [n for n in ns if n.get("type") in UTILITY_KINDS]
    save = next((n for n in ns if n.get("type") == "save_outputs"), None)
    outputs = len((save.get("data", {}) or {}).get("outputs") or {}) if save else 0
    stem_count = sum(len((n.get("data") or {}).get("stems") or []) for n in sep)
    return {
        "nodeCount": len(ns),
        "separateCount": len(sep),
        "utilCount": len(util),
    "stemCount": stem_count,
    "modelCount": len(used_models(defn)),
    "outputCount": outputs,
    }


# ==========================================================================
# Canonical graph (de)serialization — mirrors src/utils/workflowGraph.ts
# ==========================================================================
import re
from functools import cmp_to_key

ALLOWED_NODE_TYPES = {
    "input_audio", "separate", "save_outputs", "note",
    "load_audio_batch", "audio_ensemble", "audio_invert_phase", "audio_normalize",
}

DEFAULT_VIEWPORT = {"x": 0, "y": 0, "k": 1}
GRAPH_INPUT_X = 72
GRAPH_STEP_START_X = 384
GRAPH_STEP_GAP = 318
GRAPH_TOP_Y = 118
GRAPH_SAVE_GAP = 420

_STEM_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and float(v) == float(v) and abs(float(v)) != float("inf")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clone(value: Any) -> Any:
    return copy.deepcopy(value)


def _is_record(value: Any) -> bool:
    return bool(value) and isinstance(value, dict) and not isinstance(value, list)


def _read_point(value: Any) -> dict | None:
    if not _is_record(value):
        return None
    x, y = value.get("x"), value.get("y")
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return {"x": float(x), "y": float(y)}
    return None


def _read_viewport(value: Any) -> dict | None:
    pt = _read_point(value)
    if not pt:
        return None
    k = value.get("k")
    if isinstance(k, (int, float)) and _finite(k):
        return {"x": pt["x"], "y": pt["y"], "k": _clamp(float(k), 0.25, 2.5)}
    return None


def create_workflow_graph_node_id(prefix: str = "node") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def create_workflow_graph_edge_id(prefix: str = "edge") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def is_workflow_graph_definition(value: Any) -> bool:
    return _is_record(value) and value.get("kind") == KIND and _is_record(value.get("graph"))


def create_empty_workflow_graph_definition() -> dict[str, Any]:
    return {
        "version": VERSION,
        "kind": KIND,
        "defaults": {
            "device": "auto",
            "output_format": "wav",
            "model_dir": None,
            "inference_params": {"normalize": False},
        },
        "graph": {
            "viewport": dict(DEFAULT_VIEWPORT),
            "nodes": [
                {"id": "input", "type": "input_audio", "position": {"x": GRAPH_INPUT_X, "y": 210}, "data": {}},
                {"id": "save", "type": "save_outputs",
                 "position": {"x": 420 + GRAPH_STEP_GAP + GRAPH_SAVE_GAP, "y": 192}, "data": {"outputs": {}}},
            ],
            "edges": [],
        },
    }


def _parse_graph_node(value: Any) -> dict | None:
    if not _is_record(value):
        return None
    nid = str(value.get("id") or "").strip()
    ntype = str(value.get("type") or "").strip()
    position = _read_point(value.get("position"))
    if not nid or not position:
        return None
    if ntype not in ALLOWED_NODE_TYPES:
        return None
    data = value.get("data")
    return {
        "id": nid,
        "type": ntype,
        "position": position,
        "data": _clone(data) if _is_record(data) else {},
    }


def _parse_graph_edge(value: Any) -> dict | None:
    if not _is_record(value):
        return None
    eid = str(value.get("id") or "").strip() or create_workflow_graph_edge_id("edge")
    s = value.get("source") if _is_record(value.get("source")) else {}
    t = value.get("target") if _is_record(value.get("target")) else {}
    sn = str(s.get("nodeId") or "").strip()
    sp = str(s.get("portId") or "").strip()
    tn = str(t.get("nodeId") or "").strip()
    tp = str(t.get("portId") or "").strip()
    if not sn or not sp or not tn or not tp:
        return None
    return {"id": eid, "source": {"nodeId": sn, "portId": sp}, "target": {"nodeId": tn, "portId": tp}}


def ensure_graph_core_nodes(definition: dict[str, Any]) -> dict[str, Any]:
    definition = _clone(definition)
    raw = definition.get("graph", {}).get("nodes", [])
    by_id = {n["id"]: n for n in raw if _is_record(n)}
    if "input" not in by_id:
        raw.insert(0, {"id": "input", "type": "input_audio", "position": {"x": GRAPH_INPUT_X, "y": 210}, "data": {}})
    if "save" not in by_id:
        sep_count = sum(1 for n in raw if _is_record(n) and n.get("type") == "separate")
        raw.append({"id": "save", "type": "save_outputs",
                    "position": {"x": 420 + max(1, sep_count) * GRAPH_STEP_GAP + GRAPH_SAVE_GAP, "y": 192},
                    "data": {"outputs": {}}})
    definition.setdefault("graph", {})["nodes"] = raw
    return definition


def read_workflow_graph_definition(value: Any) -> dict[str, Any]:
    if not _is_record(value):
        return create_empty_workflow_graph_definition()
    if not is_workflow_graph_definition(value):
        return _migrate_legacy_workflow_to_graph(value)
    defaults = value.get("defaults") if _is_record(value.get("defaults")) else {}
    inference_defaults = defaults.get("inference_params") if _is_record(defaults.get("inference_params")) else {}
    graph = value.get("graph") if _is_record(value.get("graph")) else {}
    md = defaults.get("model_dir")
    normalized: dict[str, Any] = {
        "version": VERSION,
        "kind": KIND,
        "defaults": {
            "device": str(defaults.get("device") or "auto"),
            "output_format": str(defaults.get("output_format") or "wav"),
            "model_dir": str(md).strip() if isinstance(md, str) and str(md).strip() else None,
            "inference_params": _clone(inference_defaults),
        },
        "graph": {
            "viewport": _read_viewport(graph.get("viewport")) or dict(DEFAULT_VIEWPORT),
            "nodes": [n for n in (_parse_graph_node(x) for x in (graph.get("nodes") if isinstance(graph.get("nodes"), list) else [])) if n],
            "edges": [e for e in (_parse_graph_edge(x) for x in (graph.get("edges") if isinstance(graph.get("edges"), list) else [])) if e],
        },
    }
    return ensure_graph_core_nodes(normalized)


def serialize_workflow_graph_definition(definition: dict[str, Any]) -> dict[str, Any]:
    return _clone(ensure_graph_core_nodes(definition))


def sort_workflow_graph_step_nodes(definition: dict[str, Any]) -> list[dict[str, Any]]:
    step_nodes = [n for n in definition.get("graph", {}).get("nodes", []) if n.get("type") == "separate"]
    node_map = {n["id"]: n for n in step_nodes}
    incoming = {n["id"]: 0 for n in step_nodes}
    outgoing = {n["id"]: [] for n in step_nodes}

    for edge in definition.get("graph", {}).get("edges", []):
        t = edge.get("target") if _is_record(edge.get("target")) else {}
        if t.get("portId") != "input":
            continue
        if t.get("nodeId") not in node_map:
            continue
        s = edge.get("source") if _is_record(edge.get("source")) else {}
        if s.get("nodeId") not in node_map:
            continue
        incoming[t["nodeId"]] = incoming.get(t["nodeId"], 0) + 1
        outgoing[s["nodeId"]].append(t["nodeId"])

    def by_position(a, b):
        if a["position"]["x"] != b["position"]["x"]:
            return -1 if a["position"]["x"] < b["position"]["x"] else 1
        if a["position"]["y"] != b["position"]["y"]:
            return -1 if a["position"]["y"] < b["position"]["y"] else 1
        return -1 if a["id"] < b["id"] else (1 if a["id"] > b["id"] else 0)

    queue = sorted([n for n in step_nodes if incoming.get(n["id"], 0) == 0], key=cmp_to_key(by_position))
    ordered: list[dict[str, Any]] = []
    while queue:
        queue.sort(key=cmp_to_key(by_position))
        current = queue.pop(0)
        ordered.append(current)
        for target in outgoing.get(current["id"], []):
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(node_map[target])
    if len(ordered) == len(step_nodes):
        return ordered
    ordered_set = {n["id"] for n in ordered}
    return ordered + sorted([n for n in step_nodes if n["id"] not in ordered_set], key=cmp_to_key(by_position))


# ---- legacy migration (old steps/ui.nodeEditor shape) --------------------
def _read_legacy_steps(definition: dict[str, Any]) -> list[dict[str, Any]]:
    raw_steps = definition.get("steps") if isinstance(definition.get("steps"), list) else []
    seen: set[str] = set()
    steps: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        item = raw if _is_record(raw) else {}
        inference = item.get("inference_params") if _is_record(item.get("inference_params")) else {}
        save = item.get("save") if _is_record(item.get("save")) else {}
        cand = str(item.get("id") or "").strip() or create_workflow_graph_node_id("step")
        if cand not in seen:
            seen.add(cand)
        else:
            cand = create_workflow_graph_node_id("step")
            seen.add(cand)
        stems = item.get("stems") if isinstance(item.get("stems"), list) else None
        if stems is None:
            stems = list(save.keys())
        stems = [str(s).strip() for s in stems if str(s).strip()]
        overlap = None
        if _finite(inference.get("overlap_size")):
            overlap = inference["overlap_size"]
        raw_input = item.get("input")
        inp = str(raw_input) if raw_input else ("input" if index == 0 else "")
        steps.append({
            "id": cand,
            "model": str(item.get("model") or ""),
            "input": inp,
            "stems": stems,
            "save": {str(k): str(v or "") for k, v in (save.items() if _is_record(save) else [])},
            "overlapSize": overlap,
        })
    return steps


def _read_legacy_note(value: Any) -> dict | None:
    if not _is_record(value):
        return None
    pt = _read_point(value)
    if not pt:
        return None
    return {
        "id": str(value.get("id") or "").strip() or create_workflow_graph_node_id("note"),
        "x": pt["x"], "y": pt["y"],
        "title": str(value.get("title") or ""),
        "content": str(value.get("content") or ""),
        "color": str(value.get("color") or "amber"),
    }


def _create_default_legacy_ui(steps: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = {
        "input": {"x": GRAPH_INPUT_X, "y": 210},
        "save": {"x": 420 + max(1, len(steps)) * GRAPH_STEP_GAP, "y": 192},
    }
    for i, step in enumerate(steps):
        nodes[step["id"]] = {"x": GRAPH_STEP_START_X + i * GRAPH_STEP_GAP, "y": GRAPH_TOP_Y + (i % 2) * 96}
    return {"viewport": dict(DEFAULT_VIEWPORT), "nodes": nodes, "notes": [], "collapsedStepIds": []}


def _read_legacy_node_editor_ui(definition: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    fallback = _create_default_legacy_ui(steps)
    ui = definition.get("ui") if _is_record(definition.get("ui")) else {}
    node_editor = ui.get("nodeEditor") if _is_record(ui.get("nodeEditor")) else {}
    raw_nodes = node_editor.get("nodes") if _is_record(node_editor.get("nodes")) else {}
    nodes = dict(fallback["nodes"])
    if _is_record(raw_nodes):
        for key, value in raw_nodes.items():
            pt = _read_point(value)
            if pt:
                nodes[str(key)] = pt
    notes = []
    if isinstance(node_editor.get("notes"), list):
        for n in node_editor["notes"]:
            rn = _read_legacy_note(n)
            if rn:
                notes.append(rn)
    collapsed = node_editor.get("collapsedStepIds") if isinstance(node_editor.get("collapsedStepIds"), list) else []
    collapsed = [str(c).strip() for c in collapsed if str(c).strip()]
    return {
        "viewport": _read_viewport(node_editor.get("viewport")) or fallback["viewport"],
        "nodes": nodes,
        "notes": notes,
        "collapsedStepIds": collapsed,
    }


def _migrate_legacy_workflow_to_graph(definition: dict[str, Any]) -> dict[str, Any]:
    defaults = definition.get("defaults") if _is_record(definition.get("defaults")) else {}
    inference_defaults = defaults.get("inference_params") if _is_record(defaults.get("inference_params")) else {}
    steps = _read_legacy_steps(definition)
    ui = _read_legacy_node_editor_ui(definition, steps)
    consumed = _build_workflow_consumed_stem_set(steps)
    save_outputs: dict[str, str] = {}
    nodes: list[dict[str, Any]] = [
        {"id": "input", "type": "input_audio", "position": dict(ui["nodes"].get("input", {"x": GRAPH_INPUT_X, "y": 210})), "data": {}},
    ]
    for i, step in enumerate(steps):
        nodes.append({
            "id": step["id"], "type": "separate",
            "position": dict(ui["nodes"].get(step["id"], {"x": GRAPH_STEP_START_X, "y": GRAPH_TOP_Y})),
            "data": {"model": step["model"], "stems": list(step["stems"]), "overlapSize": step["overlapSize"],
                     "collapsed": step["id"] in ui["collapsedStepIds"]},
        })
    nodes.append({"id": "save", "type": "save_outputs",
                  "position": dict(ui["nodes"].get("save", {"x": 420 + max(1, len(steps)) * GRAPH_STEP_GAP + GRAPH_SAVE_GAP, "y": 192})),
                  "data": {"outputs": save_outputs}})
    for note in ui["notes"]:
        nodes.append({"id": note["id"], "type": "note", "position": {"x": note["x"], "y": note["y"]},
                      "data": {"title": note["title"], "content": note["content"], "color": note["color"]}})
    edges: list[dict[str, Any]] = []
    for step in steps:
        inp = str(step["input"] or "").strip()
        if not inp:
            continue
        if inp == "input":
            edges.append({"id": create_workflow_graph_edge_id("edge_input"),
                          "source": {"nodeId": "input", "portId": "audio"},
                          "target": {"nodeId": step["id"], "portId": "input"}})
            continue
        parts = inp.split(".", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        edges.append({"id": create_workflow_graph_edge_id("edge_input"),
                      "source": {"nodeId": parts[0], "portId": f"stem:{parts[1]}"},
                      "target": {"nodeId": step["id"], "portId": "input"}})
    for step in steps:
        for stem in step["stems"]:
            ref = f"{step['id']}.{stem}"
            output_dir = str(step["save"].get(stem) or "").strip() or ("" if ref in consumed else safe_workflow_stem_dir(stem))
            if not output_dir:
                continue
            save_outputs[ref] = output_dir
            edges.append({"id": create_workflow_graph_edge_id("edge_save"),
                          "source": {"nodeId": step["id"], "portId": f"stem:{stem}"},
                          "target": {"nodeId": "save", "portId": f"save:{ref}"}})
    return {
        "version": VERSION,
        "kind": KIND,
        "defaults": {
            "device": str(defaults.get("device") or "auto"),
            "output_format": str(defaults.get("output_format") or "wav"),
            "model_dir": str(defaults["model_dir"]).strip() if isinstance(defaults.get("model_dir"), str) and str(defaults["model_dir"]).strip() else None,
            "inference_params": {**_clone(inference_defaults), "normalize": bool(inference_defaults.get("normalize"))},
        },
        "graph": {"viewport": ui["viewport"], "nodes": nodes, "edges": edges},
    }


# ==========================================================================
# Draft model — mirrors src/utils/workflowDefinition.ts
# ==========================================================================
def safe_workflow_stem_dir(stem: Any) -> str:
    s = str(stem).strip()
    out = _STEM_RE.sub("_", s)
    return out or s or "stem"


def parse_model_stems(value: Any = None) -> list[str]:
    seen: set[str] = set()
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = re.split(r"[,，;；/|\n]+", str(value or ""))
    out: list[str] = []
    for item in raw_items:
        it = str(item or "").strip().strip('"\'[](){}').strip()
        if not it:
            continue
        key = it.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _read_separate_node_data(node: dict[str, Any]) -> dict[str, Any]:
    data = node.get("data") if _is_record(node.get("data")) else {}
    inference_params = data.get("inferenceParams") if _is_record(data.get("inferenceParams")) else {}
    overlap = None
    if _finite(data.get("overlapSize")):
        overlap = data["overlapSize"]
    elif _finite(inference_params.get("overlap_size")):
        overlap = inference_params["overlap_size"]
    stems = [str(s).strip() for s in (data.get("stems") if isinstance(data.get("stems"), list) else []) if str(s).strip()]
    mk = data.get("modelKind")
    custom = data.get("customModelType")
    comfy = data.get("comfyMeta") if _is_record(data.get("comfyMeta")) else None
    return {
        "model": str(data.get("model") or ""),
        "stems": stems,
        "overlapSize": overlap,
        "collapsed": bool(data.get("collapsed")),
        "modelKind": str(mk).strip() if isinstance(mk, str) and mk.strip() else None,
        "customModelType": str(custom).strip() if isinstance(custom, str) and custom.strip() else None,
        "comfyMeta": _clone(comfy) if comfy else None,
    }


def _graph_save_output_map(definition: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in definition.get("graph", {}).get("nodes", []):
        if node.get("type") == "save_outputs":
            outputs = node.get("data", {}).get("outputs") if _is_record(node.get("data")) else None
            if _is_record(outputs):
                for k, v in outputs.items():
                    v = str(v or "").strip()
                    if v:
                        out[str(k)] = v
    return out


def _normalize_save_target_source(source: Any) -> str:
    value = str(source or "").strip()
    if not value or value == "input":
        return ""
    return value


def _is_utility_node_type(type_: str) -> bool:
    return type_ in ("load_audio_batch", "audio_ensemble", "audio_invert_phase", "audio_normalize")


def _source_node_to_draft_input_value(source: dict | None, source_port_id: Any) -> str:
    if not source:
        return ""
    stype = source.get("type")
    if stype == "input_audio":
        return "input"
    if stype == "separate" and str(source_port_id).startswith("stem:"):
        return f"{source['id']}.{source_port_id[5:]}"
    if _is_utility_node_type(stype) and source_port_id == "audio":
        return f"utility:{source['id']}"
    return ""


def _graph_input_to_draft_input(definition: dict[str, Any], step_id: str) -> str:
    for edge in definition.get("graph", {}).get("edges", []):
        t = edge.get("target") if _is_record(edge.get("target")) else {}
        if t.get("nodeId") == step_id and t.get("portId") == "input":
            s = edge.get("source") if _is_record(edge.get("source")) else {}
            node = get_node(definition, str(s.get("nodeId") or ""))
            val = _source_node_to_draft_input_value(node, s.get("portId"))
            return val or "input"
    return "input"


def _graph_utility_input_map(definition: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    if not _is_utility_node_type(node.get("type")):
        return {}
    base_data = _clone(node.get("data") if _is_record(node.get("data")) else {})
    input_map: dict[str, str] = {}
    for edge in definition.get("graph", {}).get("edges", []):
        t = edge.get("target") if _is_record(edge.get("target")) else {}
        if t.get("nodeId") == node["id"]:
            s = edge.get("source") if _is_record(edge.get("source")) else {}
            source = get_node(definition, str(s.get("nodeId") or ""))
            val = _source_node_to_draft_input_value(source, s.get("portId"))
            if val:
                input_map[str(t.get("portId"))] = val
    kind = node.get("type")
    if kind == "audio_ensemble":
        input_count = max(2, min(10, int(base_data.get("inputCount") or 2) or 2))
        inputs = [str(input_map.get(f"input:{i}", "")) for i in range(input_count)]
        weights = list(base_data.get("weights"))[:input_count] if isinstance(base_data.get("weights"), list) else [1] * input_count
        return {**base_data, "inputCount": input_count, "weights": weights, "inputs": inputs}
    if kind in ("audio_invert_phase", "audio_normalize"):
        return {**base_data, "input": str(input_map.get("input", ""))}
    return base_data


def _utility_node_input_values(node: dict[str, Any]) -> list[dict[str, str]]:
    kind = node.get("kind")
    if kind == "audio_ensemble":
        input_count = max(2, min(10, int(node.get("data", {}).get("inputCount") or 2) or 2))
        raw = node.get("data", {}).get("inputs") if isinstance(node.get("data", {}).get("inputs"), list) else []
        return [{"portId": f"input:{i}", "value": str(raw[i] if i < len(raw) else "").strip()} for i in range(input_count)]
    if kind in ("audio_invert_phase", "audio_normalize"):
        return [{"portId": "input", "value": str(node.get("data", {}).get("input") or "").strip()}]
    return []


def _is_executable_graph_node_type(type_: str) -> bool:
    return type_ in ("input_audio", "separate", "load_audio_batch", "audio_ensemble", "audio_invert_phase", "audio_normalize")


def _workflow_graph_has_cycle(definition: dict[str, Any]) -> bool:
    executable_ids = {n["id"] for n in definition.get("graph", {}).get("nodes", []) if _is_executable_graph_node_type(n.get("type"))}
    outgoing = {i: [] for i in executable_ids}
    for edge in definition.get("graph", {}).get("edges", []):
        s = edge.get("source", {}).get("nodeId")
        t = edge.get("target", {}).get("nodeId")
        if s in executable_ids and t in executable_ids:
            outgoing[s].append(t)
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(nid: str) -> bool:
        if nid in visiting:
            return True
        if nid in visited:
            return False
        visiting.add(nid)
        for nxt in outgoing.get(nid, []):
            if visit(nxt):
                return True
        visiting.discard(nid)
        visited.add(nid)
        return False

    return any(visit(i) for i in executable_ids)


def _workflow_graph_source_port_is_valid(node: dict[str, Any], port_id: str) -> bool:
    ntype = node.get("type")
    if ntype == "input_audio":
        return port_id == "audio"
    if ntype == "separate":
        if not port_id.startswith("stem:"):
            return False
        stem = port_id[5:].strip().lower()
        stems = [str(s).strip().lower() for s in (node.get("data", {}).get("stems") or [])]
        return bool(stem) and stem in stems
    if _is_utility_node_type(ntype):
        return port_id == "audio"
    return False


def _workflow_graph_target_port_is_valid(node: dict[str, Any], port_id: str) -> bool:
    ntype = node.get("type")
    if ntype == "separate":
        return port_id == "input"
    if ntype == "save_outputs":
        return port_id.startswith("save:")
    if ntype == "audio_ensemble":
        if not port_id.startswith("input:"):
            return False
        try:
            idx = int(port_id.split(":", 1)[1])
        except (IndexError, ValueError):
            return False
        input_count = max(2, min(10, int(node.get("data", {}).get("inputCount") or 2) or 2))
        return isinstance(idx, int) and idx >= 0 and idx < input_count
    if ntype in ("audio_invert_phase", "audio_normalize"):
        return port_id == "input"
    return False


def _workflow_graph_edge_is_valid(definition: dict[str, Any], edge: dict[str, Any]) -> bool:
    source = next((n for n in definition.get("graph", {}).get("nodes", []) if n["id"] == edge.get("source", {}).get("nodeId")), None)
    target = next((n for n in definition.get("graph", {}).get("nodes", []) if n["id"] == edge.get("target", {}).get("nodeId")), None)
    if not source or not target:
        return False
    return _workflow_graph_source_port_is_valid(source, edge.get("source", {}).get("portId")) \
        and _workflow_graph_target_port_is_valid(target, edge.get("target", {}).get("portId"))


def _get_workflow_graph_issue_summary(definition: dict[str, Any]) -> dict[str, Any]:
    node_ids = {n["id"] for n in definition.get("graph", {}).get("nodes", [])}
    edges = definition.get("graph", {}).get("edges", [])
    dangling = sum(1 for e in edges if e.get("source", {}).get("nodeId") not in node_ids or e.get("target", {}).get("nodeId") not in node_ids)
    invalid = sum(1 for e in edges
                  if e.get("source", {}).get("nodeId") in node_ids and e.get("target", {}).get("nodeId") in node_ids
                  and not _workflow_graph_edge_is_valid(definition, e))
    valid_edges = [e for e in edges
                   if e.get("source", {}).get("nodeId") in node_ids and e.get("target", {}).get("nodeId") in node_ids
                   and _workflow_graph_edge_is_valid(definition, e)]
    incoming_counts: dict[str, int] = {}
    for e in valid_edges:
        key = f"{e['target']['nodeId']}:{e['target']['portId']}"
        incoming_counts[key] = incoming_counts.get(key, 0) + 1
    duplicate = sum(1 for c in incoming_counts.values() if c > 1)
    save_node_ids = {n["id"] for n in definition.get("graph", {}).get("nodes", []) if n.get("type") == "save_outputs"}
    save_output_count = sum(1 for e in valid_edges
                            if e.get("target", {}).get("nodeId") in save_node_ids
                            and str(e.get("target", {}).get("portId") or "").startswith("save:"))
    return {
        "danglingConnectionCount": dangling,
        "invalidConnectionCount": invalid,
        "duplicateInputConnectionCount": duplicate,
        "graphCycleDetected": _workflow_graph_has_cycle(definition),
        "saveOutputCount": save_output_count,
        "noSaveOutputs": save_output_count == 0,
    }


def _build_workflow_consumed_stem_set(steps: list[dict[str, Any]]) -> set[str]:
    consumed: set[str] = set()
    for step in steps:
        raw_input = str(step.get("input") or "").strip()
        if "." not in raw_input:
            continue
        parts = raw_input.split(".", 2)
        if not parts[0] or not parts[1]:
            continue
        consumed.add(f"{parts[0]}.{parts[1]}")
    return consumed


def _build_workflow_consumed_value_set(steps: list[dict[str, Any]], utility_nodes: list[dict[str, Any]] | None = None) -> set[str]:
    consumed: set[str] = set()

    def collect(raw_value: Any) -> None:
        value = _normalize_save_target_source(raw_value)
        if value:
            consumed.add(value)

    for step in (steps or []):
        collect(step.get("input"))
    for node in (utility_nodes or []):
        for item in _utility_node_input_values(node):
            collect(item["value"])
    return consumed


def _draft_input_value_to_graph_edge(target_node_id: str, target_port_id: str, value: Any) -> dict | None:
    inp = str(value or "").strip()
    if not inp:
        return None
    if inp == "input":
        return {"id": create_workflow_graph_edge_id("edge_input"),
                "source": {"nodeId": "input", "portId": "audio"},
                "target": {"nodeId": target_node_id, "portId": target_port_id}}
    if inp.startswith("utility:"):
        uid = inp[8:].strip()
        if not uid:
            return None
        return {"id": create_workflow_graph_edge_id("edge_utility"),
                "source": {"nodeId": uid, "portId": "audio"},
                "target": {"nodeId": target_node_id, "portId": target_port_id}}
    parts = inp.split(".", 2)
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return None
    return {"id": create_workflow_graph_edge_id("edge_input"),
            "source": {"nodeId": parts[0], "portId": f"stem:{parts[1]}"},
            "target": {"nodeId": target_node_id, "portId": target_port_id}}


# ---- id creators / drafts ------------------------------------------------
def create_workflow_step_id() -> str:
    return f"step_{uuid.uuid4().hex}"


def create_step_draft(index: int = 0) -> dict[str, Any]:
    return {
        "id": create_workflow_step_id(),
        "model": "",
        "input": "input" if index == 0 else "",
        "stems": [],
        "save": {},
        "overlapSize": None,
        "modelKind": None,
        "customModelType": None,
    }


def create_workflow_note_id() -> str:
    return f"note_{uuid.uuid4().hex}"


def create_workflow_utility_node_id(prefix: str = "tool") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def create_default_workflow_node_editor_ui(steps: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    steps = steps or [create_step_draft()]
    nodes: dict[str, Any] = {
        "input": {"x": 72, "y": 210},
        "save": {"x": 420 + max(1, len(steps)) * 318, "y": 192},
    }
    for i, step in enumerate(steps):
        nodes[step.get("id") or f"step_{i + 1}"] = {"x": 384 + i * 318, "y": 118 + (i % 2) * 96}
    return {"viewport": dict(DEFAULT_VIEWPORT), "nodes": nodes, "notes": [], "collapsedStepIds": []}


def clone_workflow_node_editor_ui(ui: dict[str, Any]) -> dict[str, Any]:
    return {
        "viewport": dict(ui.get("viewport", DEFAULT_VIEWPORT)),
        "nodes": {k: dict(v) for k, v in (ui.get("nodes") or {}).items()},
        "notes": [_clone(n) for n in (ui.get("notes") or [])],
        "collapsedStepIds": list(ui.get("collapsedStepIds") or []),
    }


def ensure_workflow_step_ids(steps: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for step in steps:
        nid = str(step.get("id") or "").strip() or create_workflow_step_id()
        if nid in seen:
            nid = create_workflow_step_id()
        step["id"] = nid
        seen.add(nid)


def get_workflow_batch_input_configs(definition: Any) -> list[dict[str, Any]]:
    draft = hydrate_workflow_definition(definition)
    out: list[dict[str, Any]] = []
    for node in (draft.get("utilityNodes") or []):
        if node.get("kind") == "load_audio_batch":
            folder = str(node.get("data", {}).get("folder") or "").strip()
            if not folder:
                continue
            sort_files = node.get("data", {}).get("sortFiles")
            out.append({
                "folder": folder,
                "recursive": bool(node.get("data", {}).get("recursive")),
                "sortFiles": True if sort_files is None else bool(sort_files),
            })
    return out


def _draft_to_graph(draft: dict[str, Any]) -> dict[str, Any]:
    ensure_workflow_step_ids(draft.get("steps") or [])
    ui = draft.get("ui") if _is_record(draft.get("ui")) else None
    ui = clone_workflow_node_editor_ui(ui) if ui else create_default_workflow_node_editor_ui(draft.get("steps") or [])
    steps = draft.get("steps") or []
    utility_nodes = draft.get("utilityNodes") or []
    consumed = _build_workflow_consumed_value_set(steps, utility_nodes)
    save_outputs: dict[str, str] = {}
    nodes: list[dict[str, Any]] = [
        {"id": "input", "type": "input_audio",
         "position": dict(ui["nodes"].get("input", {"x": GRAPH_INPUT_X, "y": 210})), "data": {}},
    ]
    for i, step in enumerate(steps):
        pos = ui["nodes"].get(step.get("id")) or {"x": GRAPH_STEP_START_X + i * GRAPH_STEP_GAP, "y": GRAPH_TOP_Y + (i % 2) * 96}
        data: dict[str, Any] = {
            "model": step.get("model", ""),
            "stems": list(step.get("stems") or []),
            "overlapSize": step.get("overlapSize"),
            "collapsed": (ui.get("collapsedStepIds") or []).count(step.get("id")) > 0,
            "modelKind": step.get("modelKind"),
            "customModelType": step.get("customModelType"),
        }
        comfy = step.get("comfyMeta")
        if _is_record(comfy):
            data["comfyMeta"] = _clone(comfy)
        nodes.append({"id": step.get("id"), "type": "separate", "position": dict(pos), "data": data})
    save_pos = ui["nodes"].get("save") or {"x": 420 + max(1, len(steps)) * GRAPH_STEP_GAP + GRAPH_SAVE_GAP, "y": 192}
    nodes.append({"id": "save", "type": "save_outputs", "position": dict(save_pos), "data": {"outputs": save_outputs}})
    for node in utility_nodes:
        nodes.append({"id": node.get("id"), "type": node.get("kind"),
                      "position": {"x": node.get("x"), "y": node.get("y")}, "data": _clone(node.get("data") or {})})
    for note in (ui.get("notes") or []):
        ndata: dict[str, Any] = {"title": note.get("title", ""), "content": note.get("content", ""), "color": note.get("color", "amber")}
        if isinstance(note.get("fontSize"), (int, float)) and note["fontSize"] > 0:
            ndata["fontSize"] = note["fontSize"]
        if isinstance(note.get("fontFamily"), str) and note["fontFamily"]:
            ndata["fontFamily"] = note["fontFamily"]
        nodes.append({"id": note.get("id"), "type": "note", "position": {"x": note.get("x"), "y": note.get("y")}, "data": ndata})
    edges: list[dict[str, Any]] = []
    for step in steps:
        edge = _draft_input_value_to_graph_edge(step.get("id"), "input", step.get("input"))
        if edge:
            edges.append(edge)
    for node in utility_nodes:
        for item in _utility_node_input_values(node):
            edge = _draft_input_value_to_graph_edge(node.get("id"), item["portId"], item["value"])
            if edge:
                edges.append(edge)
    for step in steps:
        for stem in (step.get("stems") or []):
            ref = f"{step.get('id')}.{stem}"
            if ref in consumed:
                continue
            output_dir = (step.get("save") or {}).get(stem, "").strip() or safe_workflow_stem_dir(stem)
            save_outputs[ref] = output_dir
            edges.append({"id": create_workflow_graph_edge_id("edge_save"),
                          "source": {"nodeId": step.get("id"), "portId": f"stem:{stem}"},
                          "target": {"nodeId": "save", "portId": f"save:{ref}"}})
    for node in utility_nodes:
        ref = f"utility:{node.get('id')}"
        if ref in consumed:
            continue
        target = next((t for t in (draft.get("saveTargets") or []) if t.get("source") == ref), None)
        output_dir = (target.get("outputDir") or "").strip() if target else ""
        if not output_dir:
            output_dir = safe_workflow_stem_dir(f"{node.get('kind')}_{str(node.get('id'))[-6:]}")
        save_outputs[ref] = output_dir
        edges.append({"id": create_workflow_graph_edge_id("edge_save"),
                      "source": {"nodeId": node.get("id"), "portId": "audio"},
                      "target": {"nodeId": "save", "portId": f"save:{ref}"}})
    return {
        "version": VERSION,
        "kind": KIND,
        "defaults": {
            "device": str(draft.get("defaultDevice") or "auto"),
            "output_format": str(draft.get("defaultFormat") or "wav"),
            "model_dir": None,
            "inference_params": {"normalize": bool(draft.get("defaultNormalize"))},
        },
        "graph": {
            "viewport": dict(ui.get("viewport") or DEFAULT_VIEWPORT),
            "nodes": nodes,
            "edges": edges,
        },
    }


def _graph_to_draft(definition: dict[str, Any]) -> dict[str, Any]:
    step_nodes = sort_workflow_graph_step_nodes(definition)
    save_outputs = _graph_save_output_map(definition)
    steps: list[dict[str, Any]] = []
    for node in step_nodes:
        d = _read_separate_node_data(node)
        step: dict[str, Any] = {
            "id": node["id"],
            "model": d["model"],
            "input": _graph_input_to_draft_input(definition, node["id"]),
            "stems": list(d["stems"]),
            "save": {stem: save_outputs.get(f"{node['id']}.{stem}", safe_workflow_stem_dir(stem)) for stem in d["stems"]},
            "overlapSize": d["overlapSize"],
            "modelKind": d["modelKind"],
            "customModelType": d["customModelType"],
        }
        if d["comfyMeta"]:
            step["comfyMeta"] = d["comfyMeta"]
        steps.append(step)
    if not steps:
        steps.append(create_step_draft())
    ensure_workflow_step_ids(steps)

    raw_nodes = definition.get("graph", {}).get("nodes", [])
    positions: dict[str, Any] = {n["id"]: {"x": n["position"]["x"], "y": n["position"]["y"]}
                                 for n in raw_nodes if n.get("type") != "note"}
    if "input" not in positions:
        positions["input"] = {"x": GRAPH_INPUT_X, "y": 210}
    if "save" not in positions:
        positions["save"] = {"x": 420 + max(1, len(steps)) * GRAPH_STEP_GAP + GRAPH_SAVE_GAP, "y": 192}
    for i, step in enumerate(steps):
        if step["id"] not in positions:
            positions[step["id"]] = {"x": GRAPH_STEP_START_X + i * GRAPH_STEP_GAP, "y": GRAPH_TOP_Y + (i % 2) * 96}

    notes: list[dict[str, Any]] = []
    for n in raw_nodes:
        if n.get("type") == "note":
            data = n.get("data") if _is_record(n.get("data")) else {}
            note: dict[str, Any] = {
                "id": n["id"], "x": n["position"]["x"], "y": n["position"]["y"],
                "title": str(data.get("title") or ""), "content": str(data.get("content") or ""),
                "color": str(data.get("color") or "amber"),
            }
            fs = data.get("fontSize")
            if isinstance(fs, (int, float)) and fs > 0:
                note["fontSize"] = fs
            if isinstance(data.get("fontFamily"), str) and data["fontFamily"]:
                note["fontFamily"] = data["fontFamily"]
            notes.append(note)

    utility_nodes: list[dict[str, Any]] = []
    for n in raw_nodes:
        if n.get("type") in ("load_audio_batch", "audio_ensemble", "audio_invert_phase", "audio_normalize"):
            utility_nodes.append({
                "id": n["id"], "kind": n["type"], "x": n["position"]["x"], "y": n["position"]["y"],
                "data": _graph_utility_input_map(definition, n),
            })

    save_node_ids = {n["id"] for n in raw_nodes if n.get("type") == "save_outputs"}
    save_targets: list[dict[str, Any]] = []
    for edge in definition.get("graph", {}).get("edges", []):
        t = edge.get("target") if _is_record(edge.get("target")) else {}
        if t.get("nodeId") in save_node_ids and str(t.get("portId") or "").startswith("save:"):
            s = edge.get("source") if _is_record(edge.get("source")) else {}
            source = get_node(definition, str(s.get("nodeId") or ""))
            source_val = _source_node_to_draft_input_value(source, s.get("portId")) if source else ""
            if source_val and source_val.startswith("utility:"):
                save_targets.append({
                    "source": source_val,
                    "outputDir": save_outputs.get(source_val, safe_workflow_stem_dir(f"{source.get('type')}_{str(source.get('id'))[-6:]}")),
                })

    defaults = definition.get("defaults") if _is_record(definition.get("defaults")) else {}
    inf = defaults.get("inference_params") if _is_record(defaults.get("inference_params")) else {}
    return {
        "defaultDevice": str(defaults.get("device") or "auto"),
        "defaultFormat": str(defaults.get("output_format") or "wav"),
        "defaultNormalize": bool(inf.get("normalize")),
        "steps": steps,
        "utilityNodes": utility_nodes,
        "saveTargets": save_targets,
        "ui": {
            "viewport": dict(definition.get("graph", {}).get("viewport") or DEFAULT_VIEWPORT),
            "nodes": positions,
            "notes": notes,
            "collapsedStepIds": [n["id"] for n in step_nodes if _read_separate_node_data(n).get("collapsed")],
        },
    }


def build_workflow_definition(draft: dict[str, Any]) -> dict[str, Any]:
    return serialize_workflow_graph_definition(_draft_to_graph(draft))


def hydrate_workflow_definition(definition: Any) -> dict[str, Any]:
    return _graph_to_draft(read_workflow_graph_definition(definition))


def get_workflow_validation_summary(definition: Any) -> dict[str, Any]:
    graph = read_workflow_graph_definition(definition)
    draft = _graph_to_draft(graph)
    batch_input_nodes = [n for n in (draft.get("utilityNodes") or []) if n.get("kind") == "load_audio_batch"]
    batch_input_missing = sum(1 for n in batch_input_nodes if not str(n.get("data", {}).get("folder") or "").strip())
    utility_input_missing = sum(
        sum(1 for item in _utility_node_input_values(n) if not item["value"])
        for n in (draft.get("utilityNodes") or [])
    )
    graph_issues = _get_workflow_graph_issue_summary(graph)
    return {
        "batchInputCount": len(batch_input_nodes),
        "batchInputMissingFolderCount": batch_input_missing,
        "batchInputMultipleUnsupported": len(batch_input_nodes) > 1,
        "utilityInputMissingCount": utility_input_missing,
        **graph_issues,
    }


def workflow_validation_message(summary: dict | None) -> str:
    """Return a human-readable validation error string (empty if valid)."""
    if not summary:
        return ""
    errors: list[str] = []
    _m = {
        "batchInputMultipleUnsupported": "批量输入不支持多配置",
        "batchInputMissingFolderCount": lambda: f"{summary.get('batchInputMissingFolderCount', 0)} 个批量输入缺少文件夹",
        "utilityInputMissingCount": lambda: f"{summary.get('utilityInputMissingCount', 0)} 个工具输入缺失",
        "danglingConnectionCount": lambda: f"{summary.get('danglingConnectionCount', 0)} 条悬空连接",
        "invalidConnectionCount": lambda: f"{summary.get('invalidConnectionCount', 0)} 条无效连接",
        "duplicateInputConnectionCount": lambda: f"{summary.get('duplicateInputConnectionCount', 0)} 条重复输入连接",
        "graphCycleDetected": "存在环路依赖",
        "noSaveOutputs": "没有保存输出",
    }
    for key, label in _m.items():
        val = summary.get(key)
        if val:
            errors.append(label() if callable(label) else str(label))
    return "\n".join(errors)


# ==========================================================================
# Simple / advanced mode analysis — mirrors src/utils/workflowSimple.ts
# ==========================================================================
SIMPLE_NODE_TYPES = {"input_audio", "separate", "save_outputs", "note"}


def _expected_save_sources(draft: dict[str, Any]) -> set[str]:
    consumed = {str(step.get("input") or "").strip() for step in (draft.get("steps") or [])
                if "." in str(step.get("input") or "")}
    out: set[str] = set()
    for step in (draft.get("steps") or []):
        for stem in (step.get("stems") or []):
            src = f"{step.get('id')}.{stem}"
            if src not in consumed:
                out.add(src)
    return out


def _has_canonical_input_topology(definition: Any) -> bool:
    graph = read_workflow_graph_definition(definition)
    by_id = {n["id"]: n for n in graph.get("graph", {}).get("nodes", [])}
    for node in graph.get("graph", {}).get("nodes", []):
        if node.get("type") != "separate":
            continue
        incoming = [e for e in graph.get("graph", {}).get("edges", [])
                    if e.get("target", {}).get("nodeId") == node["id"] and e.get("target", {}).get("portId") == "input"]
        if len(incoming) != 1:
            return False
        edge = incoming[0]
        src = by_id.get(edge.get("source", {}).get("nodeId"))
        if src is None:
            return False
        if src.get("type") == "input_audio":
            return edge.get("source", {}).get("portId") == "audio"
        return src.get("type") == "separate" and str(edge.get("source", {}).get("portId") or "").startswith("stem:")
    return True


def _has_canonical_save_behavior(definition: Any, expected: set[str]) -> bool:
    graph = read_workflow_graph_definition(definition)
    by_id = {n["id"]: n for n in graph.get("graph", {}).get("nodes", [])}
    save_nodes = [n for n in graph.get("graph", {}).get("nodes", []) if n.get("type") == "save_outputs"]
    if len(save_nodes) != 1:
        return False
    save_node = save_nodes[0]
    save_edges = [e for e in graph.get("graph", {}).get("edges", []) if e.get("target", {}).get("nodeId") == save_node["id"]]
    if len(save_edges) != len(expected):
        return False
    outputs = save_node.get("data", {}).get("outputs") if _is_record(save_node.get("data")) else None
    if not _is_record(outputs) or len(outputs) != len(expected):
        return False
    seen: set[str] = set()
    for edge in save_edges:
        s = edge.get("source") if _is_record(edge.get("source")) else {}
        source_node = by_id.get(s.get("nodeId"))
        if source_node is None or source_node.get("type") != "separate":
            return False
        if not str(s.get("portId") or "").startswith("stem:"):
            return False
        source = f"{source_node['id']}.{s['portId'][5:]}"
        if source not in expected or source in seen:
            return False
        if edge.get("target", {}).get("portId") != f"save:{source}":
            return False
        if not isinstance(outputs.get(source), str) or not outputs.get(source).strip():
            return False
        seen.add(source)
    return len(seen) == len(expected)


def analyze_simple_workflow(definition: Any) -> dict[str, Any]:
    graph = read_workflow_graph_definition(definition)
    draft = _graph_to_draft(graph)
    reasons: set[str] = set()
    runtime_nodes = [n for n in graph.get("graph", {}).get("nodes", []) if n.get("type") != "note"]
    if draft.get("utilityNodes"):
        reasons.add("utility_nodes")
    if any(n.get("type") not in SIMPLE_NODE_TYPES for n in runtime_nodes):
        reasons.add("unsupported_nodes")
    input_count = sum(1 for n in runtime_nodes if n.get("type") == "input_audio")
    save_count = sum(1 for n in runtime_nodes if n.get("type") == "save_outputs")
    if input_count != 1 or save_count != 1:
        reasons.add("unsupported_nodes")
    if any((step.get("modelKind") or step.get("customModelType")) for step in draft.get("steps") or []):
        reasons.add("custom_model_type")
    if any(step.get("comfyMeta") for step in draft.get("steps") or []):
        reasons.add("comfy_metadata")
    validation = get_workflow_validation_summary(definition)
    if (validation.get("batchInputMultipleUnsupported") or validation.get("batchInputMissingFolderCount")
            or validation.get("utilityInputMissingCount") or validation.get("danglingConnectionCount")
            or validation.get("invalidConnectionCount") or validation.get("duplicateInputConnectionCount")
            or validation.get("graphCycleDetected") or validation.get("noSaveOutputs")
            or not (draft.get("steps") or [])):
        reasons.add("invalid_graph")
    if not _has_canonical_input_topology(definition):
        reasons.add("invalid_graph")
    if not _has_canonical_save_behavior(definition, _expected_save_sources(draft)):
        reasons.add("custom_save_behavior")
    return {"editable": len(reasons) == 0, "reasonCodes": list(reasons)}


def resolve_workflow_open_mode(definition: Any) -> str:
    return "simple" if analyze_simple_workflow(definition)["editable"] else "advanced"


def hydrate_simple_workflow(definition: Any) -> dict[str, Any]:
    analysis = analyze_simple_workflow(definition)
    if not analysis["editable"]:
        raise ValueError("Workflow is not editable in simple mode: " + ", ".join(analysis["reasonCodes"]))
    return _graph_to_draft(read_workflow_graph_definition(definition))
