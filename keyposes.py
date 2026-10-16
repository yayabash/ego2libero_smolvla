"""Stage 3: robust keyposes (rest, grasp, release) and phase timing for every take.
Uses the precise pixel direction of the pinch point plus a known height (table plane)."""
import glob
import sys

import cv2
import matplotlib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARTON_H = float(sys.argv[1]) / 100.0            # measured carton height, cm -> m
GRASP_H = CARTON_H - 0.02                         # pinch sits ~2 cm below the carton top
REST_H = 0.01                                     # resting fingertips ~1 cm above table
TAG = 0.095
calib = np.load("data/camera_calib.npz")
K, dist = calib["K"], calib["dist"]


def camera_poses(d, t):
    rv, tv = d["tag_rvec"], d["tag_tvec"]
    good = np.flatnonzero(~np.isnan(rv[:, 0]))
    tq = np.clip(t, t[good[0]], t[good[-1]])
    R = Slerp(t[good], Rotation.from_rotvec(rv[good]))(tq).as_matrix()
    T = np.stack([np.interp(tq, t[good], tv[good, k]) for k in range(3)], axis=1)
    return R, T


def ray_to_plane(uv, R, T, h):
    """Pixel -> 3D point on the horizontal plane z = h (table coordinates)."""
    xy = cv2.undistortPoints(np.array([[uv]], np.float64), K, dist).ravel()
    C = -R.T @ T                                   # camera centre
    ray = R.T @ np.array([xy[0], xy[1], 1.0])      # viewing direction
    return C + (h - C[2]) / ray[2] * ray

from scipy.signal import savgol_filter


def anchored_path(pinch_px, mp_xyz, R, T, anchors):
    """Full path: exact pixel direction per frame, depth corrected at anchor frames.
    anchors = [(frame, true_xyz), ...] at moments with a known height."""
    n = len(pinch_px)
    ok = ~np.isnan(pinch_px[:, 0])
    px = np.stack([np.interp(np.arange(n), np.flatnonzero(ok), pinch_px[ok, k])
                   for k in range(2)], axis=1)
    C = np.einsum("nji,nj->ni", R, -T)                          # camera centres
    xy = cv2.undistortPoints(px.reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rays = np.einsum("nji,nj->ni", R, np.c_[xy, np.ones(n)])
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    s = np.einsum("ni,ni->n", mp_xyz - C, rays)                 # MediaPipe distance along ray

    frames, ratio = [], []
    for f, xyz in anchors:
        f = int(round(f))
        s_mp = np.median(s[max(f - 3, 0):f + 4])                # median of 7 frames, less noise
        frames.append(f)
        ratio.append(np.linalg.norm(xyz - C[f]) / s_mp)
    s_fixed = s * np.interp(np.arange(n), frames, ratio)
    return savgol_filter(C + s_fixed[:, None] * rays, 15, 2, axis=0)


def longest_closed(closed, fps):
    c = closed.copy()
    gap = int(0.2 * fps)                           # merge dips shorter than 0.2 s
    idx = np.flatnonzero(c)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap:
            c[a:b] = True
    edges = np.diff(np.r_[0, c.astype(int), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    k = np.argmax(ends - starts)
    return starts[k], ends[k] - 1


def median_pixel(px, frames):
    valid = np.flatnonzero(~np.isnan(px[:, 0]))
    sel = np.intersect1d(frames, valid)
    if len(sel) == 0:                              # no hand in this window: use nearest frames that have one
        centre = frames[len(frames) // 2]
        sel = valid[np.argsort(np.abs(valid - centre))[:5]]
        print(f"  note: no hand in window around frame {centre}, used nearest frames {sorted(sel)}")
    return np.median(px[sel], axis=0)


rows = []
for path_file in sorted(glob.glob("data/perception/take_*_path.npz")):
    name = path_file.split("/")[-1].replace("_path.npz", "")
    d = np.load(f"data/perception/{name}.npz")
    p = np.load(path_file)
    fps, n = float(d["fps"]), len(d["cam_pos"])
    t = np.arange(n) / fps
    R, T = camera_poses(d, t)
    pinch_px = (d["hand_px"][:, 4] + d["hand_px"][:, 8]) / 2

    g, r = longest_closed(p["closed"], fps)
    good = np.flatnonzero(~np.isnan(d["tag_rvec"][:, 0]))
    tag_dist = lambda f: np.abs(good - f).min() / fps       # seconds to the nearest real tag frame
    reliable = max(tag_dist(g), tag_dist(r)) < 0.25
    w = int(0.2 * fps)
    rest_f = np.arange(0, int(0.4 * fps))
    grasp_f = np.arange(g, min(g + w, n))
    rel_f = np.arange(max(r - w, 0), r + 1)

    rest = ray_to_plane(median_pixel(pinch_px, rest_f), R[rest_f[-1]], T[rest_f[-1]], REST_H)
    grasp = ray_to_plane(median_pixel(pinch_px, grasp_f), R[g], T[g], GRASP_H)
    release = ray_to_plane(median_pixel(pinch_px, rel_f), R[r], T[r], GRASP_H)
    end_f = np.arange(n - int(0.4 * fps), n)
    end = ray_to_plane(median_pixel(pinch_px, end_f), R[end_f[0]], T[end_f[0]], REST_H)
    path = anchored_path(pinch_px, p["pinch"], R, T,
                         [(rest_f.mean(), rest), (g, grasp), (r, release), (end_f.mean(), end)])
    lift = path[g:r + 1, 2].max()
    path_len = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()

    rows.append(dict(name=name, rest=rest, grasp=grasp, release=release, lift=lift,
                     t_grasp=g / fps, t_release=r / fps, t_end=n / fps, path=path, path_len=path_len, reliable=reliable))

    # independent check: draw the keyposes onto an early frame, before the carton moves
    cap = cv2.VideoCapture(f"data/takes/{name}.mov")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ok, frame = cap.read()
    rv0 = Rotation.from_matrix(R[5]).as_rotvec()
    for pt, col, label in [(rest, (255, 0, 0), "rest"), (grasp, (0, 255, 0), "grasp"),
                           (release, (0, 0, 255), "release")]:
        uv, _ = cv2.projectPoints(pt.reshape(1, 3), rv0, T[5], K, dist)
        if not np.all(np.isfinite(uv)):
            continue
        u, v = uv.ravel().astype(int)
        cv2.circle(frame, (u, v), 14, col, 4)
        cv2.putText(frame, label, (u + 18, v), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    cv2.imwrite(f"data/perception/{name}_keyposes.jpg", cv2.resize(frame, (960, 540)))

print(f"{'take':8} {'grasp x,y cm':>16} {'release x,y cm':>16} {'lift cm':>8} "
      f"{'path cm':>8} {'reach s':>8} {'carry s':>8}")
for k in rows:
    print(f"{k['name']:8} {str((100 * k['grasp'][:2]).round(1)):>16} "
          f"{str((100 * k['release'][:2]).round(1)):>16} {100 * k['lift']:8.1f} "
          f"{100 * k['path_len']:8.0f} {k['t_grasp']:8.2f} {k['t_release'] - k['t_grasp']:8.2f}"
          f"{k['t_grasp']:8.2f} {k['t_release'] - k['t_grasp']:8.2f}  {'ok' if k['reliable'] else 'REJECT'}")
    

rel = np.array([k["release"][:2] for k in rows]) * 100
print(f"\nrelease spread across takes (box never moved): x {rel[:, 0].std():.1f} cm, "
      f"y {rel[:, 1].std():.1f} cm   <- smaller is better")

np.savez("data/perception/keyposes.npz", **{f"{k['name']}_{f}": k[f] for k in rows
         for f in ["rest", "grasp", "release", "lift", "t_grasp", "t_release", "t_end", "path", "reliable"]})

fig, ax = plt.subplots(figsize=(7, 7))
h = TAG / 2 * 100
ax.plot([-h, h, h, -h, -h], [h, h, -h, -h, h], "k-", label="tag")
for k in rows:
    pts = np.array([k["rest"], k["grasp"], k["release"]])[:, :2] * 100
    ax.plot(*pts.T, "-", c="0.7")
    ax.scatter(*pts[0], c="b", s=30); ax.scatter(*pts[1], c="g", s=40)
    ax.scatter(*pts[2], c="r", s=40)
    ax.annotate(k["name"][-2:], pts[1])
ax.scatter([], [], c="b", label="rest"); ax.scatter([], [], c="g", label="grasp")
ax.scatter([], [], c="r", label="release")
ax.set(title="keyposes, all takes (table coordinates)", xlabel="x cm", ylabel="y cm",
       aspect="equal"); ax.legend()
fig.savefig("data/perception/keyposes_all.png", dpi=110)
print("saved data/perception/keyposes_all.png and take_XX_keyposes.jpg")