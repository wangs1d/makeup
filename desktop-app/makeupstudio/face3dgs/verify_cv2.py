"""verify_cv2 — COLMAP 几何验证的 OpenCV 回退。

背景：本机实测 COLMAP 3.11.1 / 4.2.0（无 CUDA 构建）的特征提取与暴力匹配正常，
但几何验证（two_view_geometries）全零，导致 mapper 无法初始化。本模块用
OpenCV 的 F-RANSAC 重做验证，把内点匹配与相对位姿写回数据库（config=CALIBRATED），
mapper 即可正常增量重建。实现只依赖 sqlite3 + numpy + cv2。
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import cv2
import numpy as np

MIN_INLIERS = 30
RANSAC_PX = 2.0
MAX_PAIRS = 4000


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """(w, x, y, z)。"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def verify_matches_with_cv2(db_path: str | Path,
                            min_inliers: int = MIN_INLIERS) -> int:
    """对 matches 表逐对做 F-RANSAC 验证，覆写 two_view_geometries。返回有效对数。"""
    con = sqlite3.connect(db_path)
    try:
        cam_rows = {r[0]: r[1:] for r in con.execute(
            "SELECT camera_id, model, width, height, params FROM cameras")}
        params_by_cam = {}
        for cam_id, (model, _w, _h, blob) in cam_rows.items():
            p = np.frombuffer(blob, dtype=np.float64).copy()
            if model in (0, 3):          # SIMPLE_PINHOLE / SIMPLE_RADIAL → f, cx, cy
                params_by_cam[cam_id] = p[:3]
            elif model == 1:             # PINHOLE → fx, fy, cx, cy
                params_by_cam[cam_id] = np.array([(p[0] + p[1]) / 2, p[2], p[3]])
            elif model == 8:             # OPENCV → fx, fy, cx, cy (忽略畸变)
                params_by_cam[cam_id] = np.array([(p[0] + p[1]) / 2, p[2], p[3]])
            else:
                params_by_cam[cam_id] = np.array([max(p[0], 1.0), p[1], p[2]])

        kp_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        def load_kp(image_id: int):
            if image_id not in kp_cache:
                row = con.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?",
                                  (image_id,)).fetchone()
                a = np.frombuffer(row[2], dtype=np.float32).reshape(row[0], row[1])
                kp_cache[image_id] = (a[:, 0].copy(), a[:, 1].copy())
            return kp_cache[image_id]

        image_cam = {r[0]: r[1] for r in con.execute("SELECT image_id, camera_id FROM images")}

        def pair_ids(pid: int, kmax: int = 2147483647):
            # 实测 3.11 写库格式：小 id 在高位（pid = kmax*min + max）
            hi, lo = divmod(pid, kmax)
            if hi == 0 or lo == 0 or hi >= lo:
                return None
            return hi, lo

        con.execute("DELETE FROM two_view_geometries")
        rows_out = []
        n_ok = 0
        for pid, data in con.execute("SELECT pair_id, data FROM matches"):
            ids = pair_ids(pid)
            if ids is None or len(data) < min_inliers * 8:
                continue
            i1, i2 = ids
            if i1 not in image_cam or i2 not in image_cam:
                continue
            m = np.frombuffer(data, dtype=np.uint32).reshape(-1, 2)
            x1, y1 = load_kp(i1)
            x2, y2 = load_kp(i2)
            src = np.stack([x1[m[:, 0]], y1[m[:, 0]]], axis=1)
            dst = np.stack([x2[m[:, 1]], y2[m[:, 1]]], axis=1)
            F, inl = cv2.findFundamentalMat(src, dst, cv2.FM_RANSAC,
                                            RANSAC_PX, 0.999, 20000)
            if F is None or inl is None or int(inl.sum()) < min_inliers:
                continue
            inl = inl.astype(bool).reshape(-1)
            inlier_matches = m[inl]
            F = F / (np.linalg.norm(F) + 1e-12)
            K = np.array([[params_by_cam[image_cam[i1]][0], 0, params_by_cam[image_cam[i1]][1]],
                          [0, params_by_cam[image_cam[i1]][0], params_by_cam[image_cam[i1]][2]],
                          [0, 0, 1]])
            K2 = np.array([[params_by_cam[image_cam[i2]][0], 0, params_by_cam[image_cam[i2]][1]],
                           [0, params_by_cam[image_cam[i2]][0], params_by_cam[image_cam[i2]][2]],
                           [0, 0, 1]])
            E = (K2.T @ F @ K)
            E = E / (np.linalg.norm(E) + 1e-12)
            _, R, t, _ = cv2.recoverPose(E, src[inl], dst[inl], K2, np.full(len(src[inl]), 1.0))
            q = _rot_to_quat(R)
            Fb = F.astype("<f8").tobytes()
            zeros3x3 = np.zeros((3, 3), "<f8").tobytes()
            data_u32 = inlier_matches.astype("<u4")
            rows_out.append((pid, int(len(inlier_matches)), 2,
                             data_u32.tobytes(), 2,      # config 2 = CALIBRATED
                             Fb, E.astype("<f8").tobytes(), zeros3x3,
                             q.astype("<f8").tobytes(), t.astype("<f8").tobytes()))
            n_ok += 1
            if n_ok >= MAX_PAIRS:
                break
        con.executemany("INSERT INTO two_view_geometries (pair_id, rows, cols, data, config, "
                        "F, E, H, qvec, tvec) VALUES (?,?,?,?,?,?,?,?,?,?)", rows_out)
        con.commit()
        return n_ok
    finally:
        con.close()
