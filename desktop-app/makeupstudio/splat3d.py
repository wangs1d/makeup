#!/usr/bin/env python3
"""splat3d — 高密度 3D 高斯泼溅（3DGS）生成器。

目标：用 3DGS 尽最大程度还原"生成后的妆容图"（内核 render_still 的输出）。
做法：直接复用渲染内核的烘焙管线（bake_skin / bake_makeup / 同一着色模型），
把 UV 纹理采样到 canonical 网格表面，生成稠密各向异性高斯（薄片圆盘，法线对齐）：
    · 基础层：全身皮肤底色高斯（面积加权均匀采样）；
    · 妆容细节层：在妆容 alpha>0 区域加密采样（更小 σ），保留唇渐变/眼线锐度；
    · 眼虹膜/口腔暗部点缀（对应内核 _draw_cavities）。
导出：.splat（antimatter15 格式）与 .ply（3DGS 标准格式，兼容主流查看器）。
还原度：render_splats_python() 用与 render_still 相同相机参数前向泼溅成图，
fidelity_psnr() 对比参考图给出客观指标。
"""
from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import cv2
import numpy as np

_SKILL_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "makeup-skill" / "scripts"
_REFS = _SKILL_SCRIPTS.parent / "references"


def _load_core():
    if "preview_render_core" not in globals():
        spec = importlib.util.spec_from_file_location(
            "preview_render_core", _SKILL_SCRIPTS / "preview_render.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        globals()["preview_render_core"] = mod
    return globals()["preview_render_core"]


def _quat_from_frame(t1: np.ndarray, t2: np.ndarray, n: np.ndarray) -> np.ndarray:
    """正交基 (列向量) → 四元数 (x, y, z, w)。"""
    m00, m01, m02 = t1[0], t2[0], n[0]
    m10, m11, m12 = t1[1], t2[1], n[1]
    m20, m21, m22 = t1[2], t2[2], n[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([x, y, z, w], np.float32)


class SplatCloudBuilder:
    """由妆容 layers 生成稠密 3DGS 点云。env 与内核 ENVS 同名预设。"""

    def __init__(self, obj_path: str | Path | None = None,
                 regions_json: str | Path | None = None, env: str = "neutral",
                 tex: int | None = None):
        self.core = _load_core()
        self.renderer = self.core.get_renderer(640, 538, tex=tex)
        self.renderer.set_env(env)
        self.env = env

    def set_env(self, env: str):
        self.env = env
        self.renderer.set_env(env)

    # ---------------- 主入口 ----------------

    def build(self, layers: list[dict], intensity: float = 1.0,
              n_base: int = 45000, n_makeup: int = 30000, seed: int = 7,
              n_detail2: int | None = None) -> dict:
        """生成稠密 3DGS 点云。

        高光采用"延迟"策略：构建时只烘漫反射色 + 逐点 sheen 值（cloud["sheen"]），
        Blinn-Phong 清漆项留给 render_splats_python 按真实视角计算——任意偏航角下
        唇釉/珠光的高光都随视角流动，而不是静态烙死在构建视角上。
        n_detail2：第二档超细采样数（默认 n_makeup//2），按 alpha² 加权集中于
        唇线/眼线/眉等高频区，σ 再缩一档，锐度直接决定"像不像真妆"。
        """
        core = self.core
        rng = np.random.default_rng(seed)
        model = self.renderer.model
        regions = self.renderer.regions

        # 与 render_still 完全一致的姿态与烘焙纹理（splat 路径抑制颗粒噪声：
        # 逐点采样会把纹理性噪声放大成彩色斑点，破坏还原度）
        splat_layers = []
        for l in layers:
            l2 = dict(l)
            l2["texture_strength"] = 0.0
            splat_layers.append(l2)
        layers = splat_layers
        V = model.pose_explicit(0.0, -2.0, 0.0, 0.03, 0.12)
        N = model.vertex_normals(V)
        # 缓存 key 必须含层内容：同层数不同 spec（如"全禁用"的素颜）会撞 key，
        # 拿到上一次的零纹理 → 妆容整体消失（历史 bug）
        import zlib as _zlib
        rgba, sheen = self.renderer.makeup_for(
            "splat:" + str(_zlib.crc32(json.dumps(layers, sort_keys=True).encode())),
            layers)
        skin, edge = self.renderer.skin, self.renderer.edge
        tint = self.renderer.tint
        L, view = self.renderer.L, self.renderer.view_dir

        tris = model.tris
        p0, p1, p2 = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        area = np.maximum(area, 1e-12)
        uv0, uv1, uv2 = model.uvs[tris[:, 0]], model.uvs[tris[:, 1]], model.uvs[tris[:, 2]]
        n0, n1, n2 = N[tris[:, 0]], N[tris[:, 1]], N[tris[:, 2]]

        def sample_points(sel_idx: np.ndarray, w: np.ndarray, n: int):
            """按权重 w 在选中三角面（行号）上采样 → (P, Nn, uv, tri_len)。"""
            p = w / w.sum()
            rows = sel_idx[rng.choice(len(sel_idx), size=n, p=p)]
            r1, r2 = rng.random(n), rng.random(n)
            su = np.sqrt(r1)
            b0, b1, b2 = 1 - su, su * (1 - r2), su * r2
            P = (p0[rows] * b0[:, None] + p1[rows] * b1[:, None] + p2[rows] * b2[:, None])
            Nn = n0[rows] * b0[:, None] + n1[rows] * b1[:, None] + n2[rows] * b2[:, None]
            Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
            uv = uv0[rows] * b0[:, None] + uv1[rows] * b1[:, None] + uv2[rows] * b2[:, None]
            tri_len = np.sqrt(area[rows])
            return P, Nn, uv, tri_len

        # ---- 基础皮肤层 ----
        base_sigma = float(np.sqrt(area.sum() / max(n_base, 1))) * 1.7
        P, Nn, uv, tlen = sample_points(np.arange(len(tris)), area, n_base)
        tex_x, tex_y = uv[:, 0] * (core.TEX - 1), (1 - uv[:, 1]) * (core.TEX - 1)
        skin_c = core._bilinear(skin, tex_x, tex_y)
        edge_a = core._bilinear(edge, tex_x, tex_y)[..., 0]
        col = self._shade_surface(skin_c, rgba, sheen, edge_a, Nn, L, view, tint,
                                  intensity, is_makeup=False)
        # 参考渲染的轮廓是几何硬边界（edge 羽化只调制妆容 alpha，不影响皮肤存在性）。
        # 高斯尾巴是软的：若 alpha 随 edge 衰减，脸缘会渐隐成光晕。改为
        #   · alpha 恒定（存在即覆盖，同硬光栅）；
        #   · σ 随 edge 收紧，抑制高斯尾巴越过几何边界外溢到背景。
        alpha = np.full(len(P), 0.95, np.float32)
        sig_base = base_sigma * (0.45 + 0.55 * np.clip(edge_a, 0, 1))
        parts = []
        parts.append(self._assemble(P, Nn, sig_base, col, alpha,
                                    np.zeros(len(P), np.float32)))

        # ---- 妆容细节层（alpha 加权加密采样，σ 更小保锐度；沿法线抬升避免与皮肤层 z-fighting）----
        va = core._bilinear(rgba[..., 3], model.uvs[:, 0] * (core.TEX - 1),
                            (1 - model.uvs[:, 1]) * (core.TEX - 1))[:, 0]
        tri_a = np.clip((va[tris[:, 0]] + va[tris[:, 1]] + va[tris[:, 2]]) / 3, 0, 1)
        sel = tri_a > 0.02
        if sel.any() and n_makeup > 0:
            Pm, Nm, uvm, tlenm = sample_points(np.nonzero(sel)[0], area[sel] * tri_a[sel], n_makeup)
            # 仅沿法线微量抬升避免与皮肤层同深度交叠；过大的抬升会在投影上
            # 平移眉/唇等特征（视差），直接拉低与参考渲染的对齐度
            Pm = Pm + Nm * (base_sigma * 0.35)
            tx, ty = uvm[:, 0] * (core.TEX - 1), (1 - uvm[:, 1]) * (core.TEX - 1)
            skin_m = core._bilinear(skin, tx, ty)
            mk = core._bilinear(rgba, tx, ty)
            sh_m = core._bilinear(sheen, tx, ty)[..., 0]
            edge_m = core._bilinear(edge, tx, ty)[..., 0]
            col_m, sh_pts = self._shade_surface(skin_m, mk, sh_m, edge_m, Nm, L, view, tint,
                                                intensity, is_makeup=True, sheen_map=sh_m,
                                                defer_spec=True)
            # col_m 已是"皮肤×妆容"合成后的最终色 → 近不透明绘制，避免半透明斑点
            a_m = np.full(len(Pm), 0.92, np.float32)
            parts.append(self._assemble(Pm, Nm, np.full(len(Pm), base_sigma * 0.5),
                                        col_m, a_m, sh_pts))

        # ---- 超细细节层（alpha² 加权 → 集中在唇线/眼线/眉等高频区，σ 再缩一档）----
        cnt2 = n_makeup // 2 if n_detail2 is None else int(n_detail2)
        sel2 = tri_a > 0.35
        if sel2.any() and cnt2 > 0:
            P2_, N2_, uv2, _ = sample_points(np.nonzero(sel2)[0],
                                             area[sel2] * tri_a[sel2] ** 2, cnt2)
            P2_ = P2_ + N2_ * (base_sigma * 0.5)
            tx2, ty2 = uv2[:, 0] * (core.TEX - 1), (1 - uv2[:, 1]) * (core.TEX - 1)
            mk2 = core._bilinear(rgba, tx2, ty2)
            skin2 = core._bilinear(skin, tx2, ty2)
            sh2 = core._bilinear(sheen, tx2, ty2)[..., 0]
            edge2 = core._bilinear(edge, tx2, ty2)[..., 0]
            col2, sh_pts2 = self._shade_surface(skin2, mk2, sh2, edge2, N2_, L, view, tint,
                                                intensity, is_makeup=True, sheen_map=sh2,
                                                defer_spec=True)
            a_2 = np.full(len(P2_), 0.9, np.float32)
            parts.append(self._assemble(P2_, N2_, np.full(len(P2_), base_sigma * 0.3),
                                        col2, a_2, sh_pts2))

        # 眼部/口腔暗部已包含在烘焙肤色纹理中（lips_inner / eyelid 底色），
        # 不再单独放暗盘 splat —— 单盘大 σ 会形成污渍状伪影。

        # ---- 锚点溅射层（render.type=="splat" 的层，如唇釉光斑）----
        # 参考渲染 render_still 会在网格之上再叠 build_splats() 的锚点高斯
        # （_draw_splats：屏幕空间、无光照、hex×tint）。不烘进点云的话，
        # 3DGS 图与参考图在唇部等区域必然存在系统性色差。
        import json as _json
        lm = _json.loads((_REFS / "landmark-regions.json").read_text(encoding="utf-8"))["regions"]
        for s in core.build_splats(layers, lm):
            for a in s["anchors"]:
                li = a["landmark"]
                pos = V[li]
                if "toward" in a:
                    pos = pos + (V[a["toward"]] - pos) * float(a.get("inset", 0.45))
                na = N[li] / (np.linalg.norm(N[li]) + 1e-12)
                pos = pos + na * (a.get("offset", 0.0012) / core.FACE_HEIGHT_M)
                sig = float(np.max(a["sigma"])) / core.FACE_HEIGHT_M
                col_a = core._hex_rgb(a["color"]) * tint
                alpha_a = float(a["alpha"]) * intensity
                helper = np.array([0.0, 1.0, 0.0]) if abs(na[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
                t1 = np.cross(helper, na)
                t1 /= np.linalg.norm(t1) + 1e-12
                t2 = np.cross(na, t1)
                rot_a = _quat_from_frame(t1, t2, na)[None, :]
                parts.append({
                    "xyz": pos[None, :],
                    "scale": np.array([[sig, sig, sig * 0.35]], np.float32),
                    "rot": rot_a,
                    "rgba": np.concatenate([col_a, [alpha_a]])[None, :],
                    "sheen": np.zeros(1, np.float32),
                })

        cloud = {}
        for k in ("xyz", "scale", "rot", "rgba", "sheen"):
            cloud[k] = np.concatenate([p[k] for p in parts], axis=0).astype(np.float32)
        return cloud

    # ---------------- 着色（复刻内核 _raster 的皮肤+妆容光照） ----------------

    def _shade_surface(self, skin, mk, sh_map, edge_a, Nn, L, view, tint,
                       intensity, is_makeup, sheen_map=None, defer_spec=False):
        """复刻内核 _raster 的皮肤+妆容光照。

        defer_spec=True 时不把 sheen/清漆高光烘进颜色，改为返回逐点 sheen 值
        （col, sh_pts），由 render_splats_python 按真实视角补 Blinn-Phong——
        构建视角之外的高光才能正确随视角流动。"""
        ndl = np.clip(Nn @ L, -1, 1)
        ndv = np.clip(Nn @ view, 0, 1)
        ndh_vec = L + view
        ndh_vec /= np.linalg.norm(ndh_vec)
        ndh = np.clip(Nn @ ndh_vec, 0, 1)

        wrap = 0.25
        ndl_w = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
        diff = 0.35 + 0.65 * ndl_w
        sss = (1.0 - ndl_w) * ndl_w
        col = skin * (diff + 0.35 * sss)[..., None] * tint
        col += (sss * 0.06)[..., None] * np.array([0.10, 0.02, 0.01])
        col += (ndh ** 42)[..., None] * 0.06 * tint

        if is_makeup:
            mwrap = 0.35
            mndl = np.clip((ndl + mwrap) / (1 + mwrap), 0, 1)
            mdiff = 0.55 + 0.45 * mndl
            mcol = mk[..., :3] * tint * mdiff[..., None]
            if not defer_spec:
                fres = (1.0 - ndv) ** 3
                sheen = fres * sheen_map * (0.15 + sheen_map * 0.5)
                finish_gloss = np.clip(sheen_map - 0.5, 0, 1) * 2.0
                spec = (ndh ** (60 + 60 * sheen_map)) * finish_gloss * 0.5
                mcol += (sheen + spec)[..., None] * tint
            a = np.clip(mk[..., 3] * intensity * edge_a, 0, 1)
            col = col * (1 - a[..., None]) + mcol * a[..., None]
            if defer_spec:
                return np.clip(col, 0, 1), np.clip(sheen_map, 0, 1).astype(np.float32)
        return np.clip(col, 0, 1)

    def _assemble(self, P, Nn, sigma2d, col, alpha, sheen=None):
        """薄片高斯：σxy 沿切平面、σz 压扁；旋转 = 法线对齐四元数。"""
        n = len(P)
        keep = alpha > 0.004
        P, Nn, sigma2d, col, alpha = P[keep], Nn[keep], sigma2d[keep], col[keep], alpha[keep]
        sh = sheen[keep] if sheen is not None else np.zeros(len(P), np.float32)
        if len(P) == 0:
            return {"xyz": np.zeros((0, 3)), "scale": np.zeros((0, 3)),
                    "rot": np.zeros((0, 4)), "rgba": np.zeros((0, 4)),
                    "sheen": np.zeros((0,), np.float32)}
        # 切平面基
        up = np.array([0.0, 1.0, 0.0])
        alt = np.array([1.0, 0.0, 0.0])
        helper = np.where((np.abs(Nn[:, 2]) < 0.9)[:, None], up[None, :], alt[None, :])
        t1 = np.cross(helper, Nn)
        t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
        t2 = np.cross(Nn, t1)
        rot = np.stack([_quat_from_frame(t1[i], t2[i], Nn[i]) for i in range(len(P))])
        scale = np.stack([sigma2d, sigma2d, sigma2d * 0.35], axis=1)
        rgba = np.concatenate([col, alpha[:, None]], axis=1)
        return {"xyz": P, "scale": scale, "rot": rot, "rgba": rgba, "sheen": sh}

    def _cavity_splats(self, V, N, model, regions, base_sigma):
        pts, cols, sig = [], [], []
        for eye in model.eye_centers:
            c = V[eye].mean(axis=0)
            n = N[eye].mean(axis=0)
            n /= np.linalg.norm(n) + 1e-12
            r = float(np.linalg.norm(V[eye].max(0) - V[eye].min(0))) / 2
            pts.append(c + n * 0.004)
            cols.append(np.array([0.16, 0.10, 0.10]))
            sig.append(max(r * 0.9, base_sigma * 2))
        lp = V[model.inner_idx]
        c = lp.mean(axis=0)
        n = N[model.inner_idx].mean(axis=0)
        n /= np.linalg.norm(n) + 1e-12
        rx = (lp[:, 0].max() - lp[:, 0].min()) / 2 * 0.9
        pts.append(c - n * 0.004)
        cols.append(np.array([0.13, 0.07, 0.08]))
        sig.append(max(rx, base_sigma * 2))
        pts = np.asarray(pts, np.float64)
        Nn = np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
        rot = np.stack([_quat_from_frame(np.array([1.0, 0, 0]), np.array([0.0, 1.0, 0]),
                                         np.array([0.0, 0.0, 1.0])) for _ in pts])
        scale = np.stack([np.asarray(sig), np.asarray(sig), np.full(len(sig), base_sigma * 0.3)], 1)
        rgba = np.concatenate([np.asarray(cols), np.ones((len(pts), 1))], 1)
        return {"xyz": pts, "scale": scale, "rot": rot, "rgba": rgba}

    # ---------------- 导出 ----------------

    @staticmethod
    def export_splat(cloud: dict, path: str | Path):
        """antimatter15 .splat：32B/splat = xyz f32 ×3 | scale f32 ×3 | rgba u8 ×4 | rot u8 ×4。"""
        xyz = cloud["xyz"].astype(np.float32)
        scale = cloud["scale"].astype(np.float32)
        rgba = np.clip(cloud["rgba"] * 255, 0, 255).astype(np.uint8)
        rot = np.clip(np.round(cloud["rot"] * 128 + 128), 0, 255).astype(np.uint8)
        n = len(xyz)
        buf = np.zeros(n, dtype=np.dtype([
            ("xyz", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)]))
        buf["xyz"], buf["scale"], buf["rgba"], buf["rot"] = xyz, scale, rgba, rot
        Path(path).write_bytes(buf.tobytes())

    @staticmethod
    def export_ply(cloud: dict, path: str | Path):
        """3DGS 标准 .ply（f_dc / opacity / scale log / rot wxyz），兼容 SuperSplat 等。

        cloud["sh_rest"] (n, K-1, 3) 存在时写出 f_rest_*（3DGS 通道主序约定：
        f_rest_{c·pc + (k-1)} = 第 c 通道第 k 个系数），读入端按数量反推 SH 阶数。"""
        xyz = cloud["xyz"].astype(np.float32)
        s = np.log(np.maximum(cloud["scale"], 1e-8)).astype(np.float32)
        rgba = np.clip(cloud["rgba"], 0, 1)
        f_dc = ((rgba[:, :3] - 0.5) / 0.28209479112561376).astype(np.float32)
        # 标准 logit（3DGS PLY 约定；读取端 sigmoid 严格互逆。旧式 log(a/√(1-a))
        # 往返一次会把高不透明度 splat 系统性压透明 → 脸面渗底色）
        opa = np.log(rgba[:, 3] / np.clip(1 - rgba[:, 3], 1e-6, None)).astype(np.float32)
        rot = cloud["rot"].astype(np.float32)
        rot = np.stack([rot[:, 3], rot[:, 0], rot[:, 1], rot[:, 2]], axis=1)  # wxyz
        sh_rest = cloud.get("sh_rest")
        n = len(xyz)
        n_rest = 0 if sh_rest is None else int(sh_rest.shape[1] * 3)
        fr = None
        if sh_rest is not None:
            pc = sh_rest.shape[1]
            fr = np.zeros((n, pc * 3), np.float32)
            for c in range(3):
                fr[:, c * pc:(c + 1) * pc] = sh_rest[:, :, c]
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
            + "".join(f"property float f_rest_{i}\n" for i in range(n_rest))
            + "property float opacity\n"
            "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
            "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
            "end_header\n")
        fields = [
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ] + [(f"f_rest_{i}", "<f4") for i in range(n_rest)] + [
            ("opacity", "<f4"),
            ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
            ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4")]
        arr = np.zeros(n, dtype=np.dtype(fields))
        arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        for k in range(3):
            arr[f"f_dc_{k}"] = f_dc[:, k]
            arr[f"scale_{k}"] = s[:, k]
        arr["opacity"] = opa
        for k in range(4):
            arr[f"rot_{k}"] = rot[:, k]
        if fr is not None:
            for i in range(n_rest):
                arr[f"f_rest_{i}"] = fr[:, i]
        with open(path, "wb") as f:
            f.write(header.encode("ascii"))
            f.write(arr.tobytes())


# ---------------- 还原度：CPU 前向泼溅渲染 + PSNR ----------------

def _view_dependent_spec(cloud: dict, R: np.ndarray) -> np.ndarray:
    """延迟高光：按真实视角（旋转后的法线 + 固定视线 (0,0,1) + 中性光源）逐点计算
    sheen fresnel + Blinn-Phong 清漆。cloud 无 sheen 键（用户点云）时返回 0。"""
    n = len(cloud["xyz"])
    sh = cloud.get("sheen")
    if sh is None or float(np.max(np.asarray(sh), initial=0.0)) < 1e-4:
        return np.zeros((n, 1), np.float32)
    sh = np.asarray(sh, np.float64)
    # 法线 = 四元数旋转 (0,0,1)，再随相机旋转 R
    rot = np.asarray(cloud["rot"], np.float64)
    qn = np.stack([2 * (rot[:, 0] * rot[:, 2] + rot[:, 1] * rot[:, 3]),
                   2 * (rot[:, 1] * rot[:, 2] - rot[:, 0] * rot[:, 3]),
                   1 - 2 * (rot[:, 0] ** 2 + rot[:, 1] ** 2)], axis=1)
    qn /= np.linalg.norm(qn, axis=1, keepdims=True) + 1e-12
    N = qn @ R.T
    view = np.array([0.0, 0.0, 1.0])
    L = np.array([0.15, 0.45, 0.85])
    L /= np.linalg.norm(L)
    H = L + view
    H /= np.linalg.norm(H)
    ndv = np.clip(N @ view, 0, 1)
    ndh = np.clip(N @ H, 0, 1)
    fres = (1.0 - ndv) ** 3
    sheen = fres * sh * (0.15 + sh * 0.5)
    finish_gloss = np.clip(sh - 0.5, 0, 1) * 2.0
    spec = (ndh ** (60 + 60 * sh)) * finish_gloss * 0.5
    return (sheen + spec)[..., None].astype(np.float32)


def render_splats_python(cloud: dict, w: int = 640, h: int = 538, yaw_deg: float = 0.0,
                         max_splats: int = 60000) -> np.ndarray:
    """与 render_still 相同相机参数（f=h*1.85, d=2.3）的前向泼溅。返回 BGR uint8。

    高光为延迟计算（_view_dependent_spec）：任意偏航角下唇釉/珠光随视角流动，
    与构建视角解耦。"""
    core = _load_core()
    model = core.get_renderer(w, h).model
    V = model.pose_explicit(yaw_deg, -2.0, 0.0, 0.03, 0.12)
    yaw_r = np.deg2rad(yaw_deg)
    cy, sy = np.cos(yaw_r), np.sin(yaw_r)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R = Ry @ np.array([[1, 0, 0], [0, np.cos(np.deg2rad(-2.0)), -np.sin(np.deg2rad(-2.0))],
                       [0, np.sin(np.deg2rad(-2.0)), np.cos(np.deg2rad(-2.0))]])
    xyz = cloud["xyz"] @ R.T
    scale = cloud["scale"]
    rgba = cloud["rgba"]
    spec = _view_dependent_spec(cloud, R)
    f = h * 1.85
    d = 2.3
    depth = d - xyz[:, 2]
    px = w / 2 + xyz[:, 0] * f / depth
    py = h / 2 - xyz[:, 1] * f / depth

    order = np.argsort(-depth)                 # 后 → 前
    if max_splats and max_splats < len(order):
        # 超出预算时丢弃最远的 splat（队列头部是最远的），保住近处细节
        order = order[-max_splats:]
    canvas = core.get_renderer(w, h)._background(2.6)
    for i in order:
        a = rgba[i, 3]
        if a < 0.01 or depth[i] < 0.2:
            continue
        sig_px = float(scale[i, 0]) * f / depth[i]
        if sig_px < 0.4:
            continue
        cx, cyy = px[i], py[i]
        half = int(min(sig_px * 3, w))
        x0, x1 = int(cx) - half, int(cx) + half
        y0, y1 = int(cyy) - half, int(cyy) + half
        if x1 < 0 or y1 < 0 or x0 >= w or y0 >= h:
            continue
        rows, cols = np.mgrid[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)]
        g = np.exp(-0.5 * (((cols - cx) / sig_px) ** 2 + ((rows - cyy) / sig_px) ** 2)) * a
        sl = canvas[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)]
        sl[:] = sl * (1 - g[..., None]) + (rgba[i, :3] + spec[i]) * g[..., None]
    return np.clip(canvas * 255, 0, 255).astype(np.uint8)[..., ::-1]


def fidelity_psnr(img_a_bgr: np.ndarray, img_b_bgr: np.ndarray) -> float:
    a = img_a_bgr.astype(np.float64) / 255.0
    b = img_b_bgr.astype(np.float64) / 255.0
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-9 else 10.0 * np.log10(1.0 / mse)
