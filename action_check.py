"""How jittery are the action labels? Clean vs noisy episodes."""
import csv

import matplotlib
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ds = LeRobotDataset("local/hmnd_ego_orange_juice", root="data/lerobot_dataset")
cols = ds.hf_dataset.with_format("numpy")
a = np.stack(cols["action"]).astype(np.float32)
ep = np.asarray(cols["episode_index"]).ravel()

rows = [r for r in csv.DictReader(open("data/replay_log.csv")) if r["saved"] == "1"]
noisy = np.array([int(r["noisy"]) for r in rows])
assert len(noisy) == ep.max() + 1, "episode count does not match replay_log.csv"

jerk = np.array([np.abs(np.diff(a[ep == e], axis=0)).mean(0) for e in range(len(noisy))])
print("mean step-to-step change of the action label")
for name, m in [("clean", noisy == 0), ("noisy", noisy == 1)]:
    print(f"  {name}: {m.sum():2d} episodes | position {jerk[m, :3].mean():.3f} | "
          f"rotation {jerk[m, 3:6].mean():.3f} | gripper {jerk[m, 6].mean():.3f}")

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for k, name in enumerate(["clean", "noisy"]):
    e = np.flatnonzero(noisy == k)[0]
    x = a[ep == e]
    for d, lab in enumerate(["x", "y", "z", "rx", "ry", "rz"]):
        ax[k].plot(x[:, d], label=lab, lw=1)
    ax[k].set_title(f"{name} episode {e}"); ax[k].set_ylabel("action")
ax[0].legend(ncol=6); ax[1].set_xlabel("step")
fig.tight_layout(); fig.savefig("action_check.png", dpi=110)
print("saved action_check.png")