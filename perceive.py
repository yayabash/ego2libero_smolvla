"""Stage 1: find the table tag and the hand in every frame of one take."""
import pathlib
import sys
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

take = pathlib.Path(sys.argv[1])
TAG = float(sys.argv[2]) / 1000.0                  # tag side in metres
calib = np.load("data/camera_calib.npz")
K, dist = calib["K"], calib["dist"]

model = pathlib.Path("data/hand_landmarker.task")
if not model.exists():
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task", model)

hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(model)),
    running_mode=vision.RunningMode.VIDEO, num_hands=1))
aruco = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
h = TAG / 2
tag_corners_3d = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)

cap = cv2.VideoCapture(str(take))
fps = cap.get(cv2.CAP_PROP_FPS)
out_dir = pathlib.Path("data/perception")
out_dir.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(out_dir / f"{take.stem}_check.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (960, 540))

rvecs, tvecs, cam_pos, hand_px, hand_3d = [], [], [], [], []
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    H, W = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    r, t, c = np.full(3, np.nan), np.full(3, np.nan), np.full(3, np.nan)
    px, xyz = np.full((21, 2), np.nan), np.full((21, 3), np.nan)

    # tag -> where the camera is, in the tag's coordinates
    corners, ids, _ = aruco.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is not None and 0 in ids.flatten():
        k = list(ids.flatten()).index(0)
        found, rv, tv = cv2.solvePnP(tag_corners_3d, corners[k].reshape(4, 2), K, dist,
                                     flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if found:
            r, t = rv.ravel(), tv.ravel()
            c = -cv2.Rodrigues(rv)[0].T @ t
            cv2.drawFrameAxes(frame, K, dist, rv, tv, TAG)

    # hand -> 21 keypoints in pixels, and the hand's own 3D shape in metres
    res = hands.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                 int(i * 1000 / fps))
    if res.hand_landmarks:
        px = np.array([[p.x * W, p.y * H] for p in res.hand_landmarks[0]])
        xyz = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]])
        for x, y in px:
            cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 0), -1)

    rvecs.append(r); tvecs.append(t); cam_pos.append(c); hand_px.append(px); hand_3d.append(xyz)
    writer.write(cv2.resize(frame, (960, 540)))
    i += 1
writer.release()

cam_pos, tvecs, hand_px = np.array(cam_pos), np.array(tvecs), np.array(hand_px)
np.savez(out_dir / f"{take.stem}.npz", fps=fps, tag_rvec=np.array(rvecs), tag_tvec=tvecs,
         cam_pos=cam_pos, hand_px=hand_px, hand_3d=np.array(hand_3d))

tag_ok, hand_ok = ~np.isnan(cam_pos[:, 0]), ~np.isnan(hand_px[:, 0, 0])
print(f"{take.name}: {i} frames at {fps:.2f} fps")
print(f"tag found:  {100 * tag_ok.mean():.0f}% of frames")
print(f"hand found: {100 * hand_ok.mean():.0f}% of frames")
dist_cm = 100 * np.linalg.norm(tvecs[tag_ok], axis=1)
print(f"camera-to-tag distance: median {np.median(dist_cm):.1f} cm "
      f"(min {dist_cm.min():.1f}, max {dist_cm.max():.1f})")
print(f"camera height above table: median {100 * np.median(cam_pos[tag_ok, 2]):.1f} cm")