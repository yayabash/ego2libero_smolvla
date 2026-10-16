"""Stage 2: turn per-frame detections into a smooth 3D pinch path in table (tag) coordinates."""
import sys

import cv2
import matplotlib
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

matplotlib.use("Agg")
import matplotlib.pyplot as plt

name = sys.argv[1]                                    # e.g. take_01
TAG = float(sys.argv[2]) / 1000.0 if len(sys.argv) > 2 else 0.095
d = np.load(f"data/perception/{name}.npz")
calib = np.load("data/camera_calib.npz")
K, dist = calib["K"], calib["dist"]
fps = float(d["fps"])
n = len(d["cam_pos"])
t = np.arange(n) / fps

# 1) camera pose for every frame, gaps filled by blending between good frames
rv, tv = d["tag_rvec"], d["tag_tvec"]
good = np.flatnonzero(~np.isnan(rv[:, 0]))
tq = np.clip(t, t[good[0]], t[good[-1]])
R_cam = Slerp(t[good], Rotation.from_rotvec(rv[good]))(tq).as_matrix()   # tag -> camera
T_cam = np.stack([np.interp(tq, t[good], tv[good, k]) for k in range(3)], axis=1)
longest_gap = (np.diff(good).max() - 1) / fps if len(good) > 1 else 0.0

# 2) + 3) hand in camera frame, then in table frame
W, P = d["hand_3d"], d["hand_px"]
pinch = np.full((n, 3), np.nan)
aperture = np.full(n, np.nan)
for i in range(n):
    if np.isnan(P[i, 0, 0]):
        continue
    found, rvh, tvh = cv2.solvePnP(W[i].astype(np.float64), P[i].astype(np.float64),
                                   K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not found:
        continue
    cam = (cv2.Rodrigues(rvh)[0] @ W[i].T).T + tvh.ravel()      # 21 points, camera frame
    tab = (R_cam[i].T @ (cam - T_cam[i]).T).T                   # 21 points, table frame
    pinch[i] = (tab[4] + tab[8]) / 2                            # thumb tip + index tip
    aperture[i] = np.linalg.norm(W[i, 4] - W[i, 8])

# 4) fill short hand gaps and smooth
def fill(x):
    m = ~np.isnan(x)
    return np.interp(np.arange(len(x)), np.flatnonzero(m), x[m])

pinch = savgol_filter(np.stack([fill(pinch[:, k]) for k in range(3)], axis=1), 11, 2, axis=0)
aperture = savgol_filter(fill(aperture), 11, 2)

# 5) height check: the resting hand at the start should sit on the table
rest = int(0.5 * fps)
raw_rest_z = np.median(pinch[:rest, 2])
pinch[:, 2] -= raw_rest_z - 0.01                                # fingertips ~1 cm above table

# rough grasp guess for the plot: aperture below the midpoint of its range
lo, hi = np.percentile(aperture, [10, 90])
closed = aperture < (lo + hi) / 2

np.savez(f"data/perception/{name}_path.npz", t=t, fps=fps, pinch=pinch,
         aperture=aperture, closed=closed)

print(f"{name}: {n / fps:.1f} s, longest tag gap {longest_gap:.2f} s")
print(f"raw resting height before correction: {100 * raw_rest_z:+.1f} cm  (expect -3 to +5)")
print(f"start (x, y): {100 * pinch[0, :2].round(3)} cm   end: {100 * pinch[-1, :2].round(3)} cm")
print(f"highest point: {100 * pinch[:, 2].max():.1f} cm above table")
print(f"path length: {100 * np.linalg.norm(np.diff(pinch, axis=0), axis=1).sum():.0f} cm")
print(f"aperture: {100 * lo:.1f} to {100 * hi:.1f} cm, closed for {closed.mean() * 100:.0f}% of the take")

# plot: top view, height over time, aperture over time
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
h = TAG / 2 * 100
ax[0].plot([-h, h, h, -h, -h], [h, h, -h, -h, h], "k-", label="tag")
sc = ax[0].scatter(pinch[:, 0] * 100, pinch[:, 1] * 100, c=t, s=6, cmap="viridis")
ax[0].scatter(*(pinch[closed, :2].T * 100), s=14, facecolors="none", edgecolors="r",
              label="closed")
ax[0].set(title="top view (table)", xlabel="x cm", ylabel="y cm", aspect="equal")
ax[0].legend(); fig.colorbar(sc, ax=ax[0], label="time s")
ax[1].plot(t, pinch[:, 2] * 100); ax[1].set(title="height above table", xlabel="s", ylabel="cm")
ax[2].plot(t, aperture * 100); ax[2].axhline((lo + hi) / 2 * 100, ls="--", c="r")
ax[2].set(title="thumb-index distance", xlabel="s", ylabel="cm")
for a in ax[1:]:
    a.fill_between(t, 0, 1, where=closed, color="r", alpha=0.1, transform=a.get_xaxis_transform())
fig.tight_layout()
fig.savefig(f"data/perception/{name}_path.png", dpi=110)
print(f"saved data/perception/{name}_path.png")