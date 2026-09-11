"""parse_look 规范化 / bake_assets 烘焙 / apply_spec 资产选择的单元测试。"""
from __future__ import annotations

import json

import pytest

import bake_assets
import parse_look


# ---------------- parse_look ----------------

def test_normalize_layer_full():
    layer = {"region": "eyeshadow", "side": "both", "opacity": 1.4, "finish": "gloss",
             "color_stops": [{"at": 0.2, "hex": "aabbcc"}, {"at": 0.8, "hex": "#DDEEFF"}],
             "texture_strength": 0.5, "shape": {"spread": 0.6}, "notes": "晕染"}
    out = parse_look.normalize_layer(layer, set())
    assert out["id"] == "eyeshadow-both"
    assert out["opacity"] == 1.0
    assert out["finish"] == "gloss"
    assert out["render"]["type"] == "mesh"        # 非唇部 gloss 也走 mesh
    assert [s["hex"] for s in out["color_stops"]] == ["#AABBCC", "#DDEEFF"]
    assert out["shape"] == {"spread": 0.6} and out["notes"] == "晕染"


def test_normalize_layer_lip_gloss_gets_splat():
    layer = {"region": "lipstick", "finish": "gloss", "opacity": 0.8,
             "color_stops": [{"at": 0, "hex": "#BB5566"}]}
    out = parse_look.normalize_layer(layer, set())
    assert out["render"]["type"] == "splat"
    assert len(out["color_stops"]) == 2           # 单色标补成两段


def test_normalize_layer_rejects_bad_region_and_dup():
    assert parse_look.normalize_layer({"region": "nope"}, set()) is None
    l = {"region": "blush", "side": "both"}
    assert parse_look.normalize_layer(l, {("blush", "both")}) is None


def test_default_steps_grouping():
    layers = [dict(id=f"{r}-both", region=r, side="both", enabled=True)
              for r in ("foundation", "concealer", "eyeshadow", "lipstick")]
    steps = parse_look.default_steps(layers)
    assert [s["area"] for s in steps] == ["base", "eye", "lip"]
    assert steps[0]["regions"] == ["foundation", "concealer"]


def test_build_spec_end_to_end():
    merged = {
        "style": "测试风格", "overall_intensity": 0.7, "confidence": 0.9,
        "layers": [
            {"region": "lipstick", "side": "both", "opacity": 0.9, "finish": "satin",
             "color_stops": [{"at": 0, "hex": "#BB5566"}]},
            {"region": "eyeshadow", "side": "both", "opacity": 0.5, "finish": "matte",
             "color_stops": [{"at": 0, "hex": "#C8A080"}]},
        ],
    }
    spec = parse_look.build_spec(merged, 3, "测试素材", None)
    assert spec["spec_version"] == "1.0"
    assert spec["name"].startswith("parsed-look")
    assert [l["region"] for l in spec["layers"]] == ["eyeshadow", "lipstick"]   # 按化妆顺序排
    assert spec["source"]["frame_count"] == 3
    assert spec["steps"][0]["area"] == "base" or spec["steps"][-1]["area"] == "lip"
    # --focus 后浓度按层数微调且仍合法
    spec2 = parse_look.build_spec(merged, 1, "x.png", ["lipstick"])
    assert [l["region"] for l in spec2["layers"]] == ["lipstick"]
    assert 0 <= spec2["intensity"] <= 1


# ---------------- bake_assets ----------------

PRESET = {
    "spec_version": "1.0", "name": "t-look", "description": "t", "intensity": 0.8,
    "layers": [
        {"id": "lipstick-both", "region": "lipstick", "side": "both", "enabled": True,
         "opacity": 0.85, "finish": "gloss", "texture_strength": 0.1,
         "color_stops": [{"at": 0, "hex": "#C9776F"}, {"at": 1, "hex": "#B85F62"}],
         "render": {"type": "splat", "splat": {"thickness": 0.0015, "density": 0.8}}},
        {"id": "blush-both", "region": "blush", "side": "both", "enabled": True,
         "opacity": 0.6, "finish": "matte", "texture_strength": 0.4,
         "color_stops": [{"at": 0, "hex": "#D89A8A"}, {"at": 1, "hex": "#C08070"}],
         "render": {"type": "splat", "splat": {"thickness": 0.0012, "density": 0.6}}},
        {"id": "foundation-both", "region": "foundation", "side": "both", "enabled": True,
         "opacity": 0.5, "finish": "satin", "texture_strength": 0.3,
         "color_stops": [{"at": 0, "hex": "#E8C8B0"}],
         "render": {"type": "splat"}},          # 不支持的 splat region → 显式跳过
    ],
}


def test_bake_creates_assets_and_shared_splat_rules(tmp_path):
    out = bake_assets.bake(PRESET, tmp_path)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    baked = json.loads((out / "baked_spec.json").read_text(encoding="utf-8"))
    assert (out / "manifest.json").exists()
    for name in manifest["files"]:
        assert (out / name).stat().st_size > 0
    # lipstick → splat 且带 toward/inset；blush splat 现已支持
    splats = {s["region"]: s for s in baked["splat_layers"]}
    assert "lipstick" in splats and "blush" in splats
    outer = [a for a in splats["lipstick"]["anchors"] if a["group"] == "lips_outer"]
    assert outer and all("toward" in a for a in outer)
    assert any(a["group"].startswith("blush_") for a in splats["blush"]["anchors"])
    # 层带 baked 引用（ramp/grain）
    bl = {l["id"]: l for l in baked["layers"]}
    assert bl["lipstick-both"]["baked"]["ramp"] in manifest["files"]
    assert bl["blush-both"]["baked"]["grain"] in manifest["files"]
    # foundation 的 splat 请求被丢弃（不支持）→ 只有 2 个 splat 层
    assert len(baked["splat_layers"]) == 2


def test_pack_zip(tmp_path):
    out = bake_assets.bake(PRESET, tmp_path)
    z = bake_assets.pack(out, "t-look")
    import zipfile
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
    assert "baked_spec.json" in names and any(n.endswith(".png") for n in names)


# ---------------- apply_spec（纯函数部分） ----------------

def test_apply_spec_load_spec_version_and_layers(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"spec_version": "1.0", "name": "x", "layers": [{"region": "blush"}]}),
                 encoding="utf-8")
    spec = __import__("apply_spec").load_spec(str(p))
    assert spec["name"] == "x"
    bad = tmp_path / "b.json"
    bad.write_text(json.dumps({"spec_version": "2.0", "layers": [{"region": "blush"}]}),
                   encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        __import__("apply_spec").load_spec(str(bad))
    assert e.value.code == 1


def test_apply_spec_collect_assets_manifest(tmp_path):
    d = tmp_path / "baked"
    d.mkdir()
    (d / "ramp_a.png").write_bytes(b"\x89PNG-fake")
    (d / "manifest.json").write_text(json.dumps({"files": ["ramp_a.png"]}), encoding="utf-8")
    assets = __import__("apply_spec").collect_assets(str(d))
    assert assets["ramp_a.png"] == b"\x89PNG-fake"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as e:
        __import__("apply_spec").collect_assets(str(empty))
    assert e.value.code == 1
