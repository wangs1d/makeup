#!/usr/bin/env python3
"""人脸追踪 sidecar：人脸关键点 + 6DoF 姿态 → UDP 发给试妆 App。

输入源（三选一）：
    --camera N   本地摄像头 + MediaPipe FaceMesh（默认）
    --relay      不开摄像头：接收试妆 App 经 UDP 中继的 JPEG 帧（单摄像头闭环，
                 帧与关键点同源，姿态对齐才精确）。端口 8767。
    --synthetic  无摄像头无 mediapipe：动画 canonical 网格自产关键点（联调/演示用）

输出协议（unity-app/tools/tracking_protocol.py 为准，UdpLandmarkReceiver.cs 按此解析）：
    v2 二进制（默认）：header + 4×4 姿态矩阵（厘米，OpenGL 相机系）+ 头部局部关键点 + 图像坐标
    --legacy-json    旧版 v1 JSON（仅 468 归一化点，App 退回固定平面映射，无姿态对齐）

用法：
    pip install mediapipe opencv-python numpy
    python face_tracker.py [--camera 0] [--port 8766] [--flip] [--smooth]
    python face_tracker.py --relay
    python face_tracker.py --synthetic --show
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from tracking_protocol import (  # noqa: E402
    DEFAULT_FOCAL_RATIO, DEFAULT_RELAY_PORT, DEFAULT_TRACK_PORT, FLAG_SMOOTHED,
    FrameAssembler, OneEuroFilter, TrackPacket, chunk_frame, is_v2, load_canonical,
    default_canonical_path, pack_v1, pack_v2, project_canonical, solve_pose)

try:
    import mediapipe as mp
except ImportError:
    mp = None

REFINE = dict(refine_landmarks=False)  # 468 点拓扑；若改 True(478 点) App 端取前 468 点


# ---------------- MediaPipe 推理 ----------------

class LandmarkModel:
    """MediaPipe FaceMesh 封装；不可用时 is_available()=False（synthetic 模式替代）。"""

    def __init__(self) -> None:
        if mp is None:
            raise SystemExit("缺少依赖：pip install mediapipe opencv-python numpy"
                             "（无 mediapipe 环境可用 --synthetic 联调）")
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, min_detection_confidence=0.5,
            min_tracking_confidence=0.5, **REFINE)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        return np.array([[p.x, p.y, p.z] for p in lm], np.float64)


class SyntheticModel:
    """合成人脸：动画 canonical（转头/张嘴/微笑）后透视投影 → MediaPipe 风格归一化点。

    用于无摄像头环境联调 App 的姿态对齐与渲染链路（同一个 solve 管线，真实可测）。
    """

    def __init__(self, w: int, h: int, focal_px: float) -> None:
        self.canonical = load_canonical(default_canonical_path())
        self.w, self.h, self.focal = w, h, focal_px
        # 表情动作的关键点组
        self.inner = np.arange(468)
        self._t0 = time.monotonic()

    def _rot(self, t: float) -> np.ndarray:
        yaw = np.deg2rad(22 * np.sin(2 * np.pi * t / 11.0))
        pitch = np.deg2rad(7 * np.sin(2 * np.pi * t / 17.0 + 1.0))
        roll = np.deg2rad(4 * np.sin(2 * np.pi * t / 23.0))
        cy, sy, cx, sx, cz, sz = (np.cos(yaw), np.sin(yaw), np.cos(pitch),
                                  np.sin(pitch), np.cos(roll), np.sin(roll))
        return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
                @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
                @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]))

    def process(self, frame_bgr: np.ndarray | None) -> np.ndarray | None:
        t = time.monotonic() - self._t0
        h, w = self.h, self.w
        if frame_bgr is not None:
            h, w = frame_bgr.shape[:2]
        mouth = max(0.0, float(np.sin(2 * np.pi * t / 7.0))) ** 2 * 0.6
        V = self.canonical.copy()
        lips = (V[:, 1] < -5.6) & (V[:, 0] > -2.2) & (V[:, 0] < 2.2) & (V[:, 2] > 3.0)
        upper = lips & (V[:, 1] > V[lips][:, 1].mean())
        V[upper, 1] += mouth * 0.5
        V[lips & ~upper, 1] -= mouth * 0.6
        depth = 48.0 + 3.0 * np.sin(2 * np.pi * t / 30.0)
        return project_canonical(V, self._rot(t), np.array([1.5, -1.0, -depth]),
                                 w, h, self.focal)


# ---------------- sidecar 主循环 ----------------

def main() -> None:
    ap = argparse.ArgumentParser(description="人脸关键点 + 姿态 UDP sidecar")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="本地摄像头编号（默认 0）")
    src.add_argument("--relay", action="store_true",
                     help=f"接收 App 中继帧（UDP :{DEFAULT_RELAY_PORT}），单摄像头闭环")
    src.add_argument("--synthetic", action="store_true", help="合成人脸（无摄像头/无 mediapipe）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_TRACK_PORT)
    ap.add_argument("--relay-port", type=int, default=DEFAULT_RELAY_PORT)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--focal-ratio", type=float, default=DEFAULT_FOCAL_RATIO,
                    help="焦距 = 比例 × 帧宽（决定 App 相机 FOV，须与真实摄像头视场接近）")
    ap.add_argument("--flip", action="store_true", help="水平镜像后再发（App 端镜像显示时用其一即可）")
    ap.add_argument("--smooth", action="store_true", help="sidecar 侧 One Euro 平滑（App 端默认自带）")
    ap.add_argument("--no-pose", action="store_true", help="不发姿态矩阵（App 退回固定平面映射）")
    ap.add_argument("--legacy-json", action="store_true", help="发旧版 v1 JSON（兼容老 App）")
    ap.add_argument("--show", action="store_true", help="弹出预览窗口（调试用）")
    ap.add_argument("--fps-limit", type=float, default=0, help="限流（0=不限）")
    args = ap.parse_args()

    canonical = load_canonical(default_canonical_path())
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)
    focal_px = args.focal_ratio * args.width

    # 输入源
    assembler = FrameAssembler()
    model: LandmarkModel | SyntheticModel
    cap = None
    if args.synthetic:
        model = SyntheticModel(args.width, args.height, focal_px)
        print("输入源：synthetic（合成人脸动画）")
    elif args.relay:
        if mp is None:
            raise SystemExit("缺少依赖：pip install mediapipe opencv-python numpy")
        model = LandmarkModel()
        relay = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        relay.bind(("127.0.0.1", args.relay_port))
        relay.settimeout(2.0)
        print(f"输入源：relay udp://127.0.0.1:{args.relay_port}（等待 App 发帧）")
    else:
        if mp is None:
            raise SystemExit("缺少依赖：pip install mediapipe opencv-python numpy")
        model = LandmarkModel()
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform == "win32" else args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        if not cap.isOpened():
            print(f"摄像头 {args.camera} 打开失败（试试 --relay 或 --synthetic）", file=sys.stderr)
            sys.exit(2)
        print(f"输入源：camera {args.camera}")

    filt = OneEuroFilter(min_cutoff=1.4, beta=0.05) if args.smooth else None
    print(f"face_tracker → udp://{args.host}:{args.port}"
          f"（{'binary v2' if not args.legacy_json else 'legacy json'}，focal≈{focal_px:.0f}px，Ctrl+C 退出）")

    seq = 0
    frames = faces = pose_fail = 0
    t0 = time.monotonic()
    last_send = 0.0
    last_face = 0.0

    def send(img: np.ndarray | None, w: int, h: int) -> None:
        nonlocal seq, faces, pose_fail, last_send, last_face
        now = time.monotonic()
        if img is None:
            if args.legacy_json:
                data = pack_v1(False, w, h, now, None)
            else:
                data = pack_v2(TrackPacket(seq=seq, t=now, w=w, h=h, focal_px=focal_px,
                                           face_ok=False))
            seq += 1
            if len(data) < 60000:
                sock.sendto(data, addr)
            last_send = now
            return

        faces += 1
        local = img_pts_out = None
        pose = None
        if not args.legacy_json and not args.no_pose:
            try:
                pose, local = solve_pose(img, canonical, w, h, focal_px)
            except np.linalg.LinAlgError:
                pose_fail += 1
        if args.legacy_json:
            data = pack_v1(True, w, h, now, img)
        else:
            pts = img
            if filt is not None and local is not None:
                local = filt(local, now)
            data = pack_v2(TrackPacket(seq=seq, t=now, w=w, h=h, focal_px=focal_px,
                                       face_ok=True, pose=None if pose is None else pose.matrix(),
                                       local=local, img=img, smoothed=filt is not None))
        seq += 1
        last_face = now
        if len(data) < 60000:
            sock.sendto(data, addr)
        last_send = now

    def overlay(frame: np.ndarray, img: np.ndarray) -> None:
        h, w = frame.shape[:2]
        for x, y in img[:, :2]:
            cv2.circle(frame, (int(x * w), int(y * h)), 1, (0, 255, 128), -1)

    try:
        while True:
            if args.fps_limit:
                gap = 1.0 / args.fps_limit - (time.monotonic() - last_send)
                if gap > 0:
                    time.sleep(gap)
            frame = img = None
            w, h = args.width, args.height

            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.05)
                    continue
                if args.flip:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                img = model.process(frame)
            elif args.relay:
                try:
                    data = relay.recv(70000)
                except socket.timeout:
                    if time.monotonic() - last_face > 1.0 and seq % 15 == 0:
                        send(None, w, h)   # 心跳：让 App 知道 sidecar 活着但没帧
                    continue
                got = assembler.feed(data) if not is_v2(data) else None
                if got is None:
                    continue
                _fid, w, h, jpeg = got
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                img = model.process(frame)
            else:  # synthetic
                img = model.process(None)

            send(img, w, h)

            if args.show and frame is not None and img is not None:
                overlay(frame, img)
                cv2.imshow("face_tracker (q=quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frames += 1
            if frames % 150 == 0:
                dt = time.monotonic() - t0
                print(f"  已发 {frames} 帧（检出 {faces}，姿态失败 {pose_fail}，"
                      f"{frames / max(dt, 1e-6):.1f} fps）")
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        if args.show:
            cv2.destroyAllWindows()
        print("face_tracker 退出")


if __name__ == "__main__":
    main()
