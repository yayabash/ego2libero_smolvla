"""Diagnose LIBERO replays: classify why a replay fails, with a labelled video and a trace plot."""
import argparse
import csv
import os

import cv2
import imageio
import matplotlib
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as Rot

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--take")
ap.add_argument("--layout", type=int)
ap.add_argument("--failed", action="store_true", help="every clean failure in data/replay_log.csv")
ap.add_argument("--compare", type=int, default=3, help="also run this many clean successes")
ap.add_argument("--lift_min", type=float, default=0.17)
ap.add_argument("--lift_max", type=float, default=0.20)
ap.add_argument("--tag", default="", help="suffix for output file names")
args = ap.parse_args()

# same settings as build_dataset.py, so replays are identical (clean replays are deterministic)
HZ, STEP_M, TASK_ID = 20, 0.0098, 9
V_MAX, MAX_STEPS = 0.18, 230
OPEN, CLOSE = -1.0, 1.0
GRASP_DZ, RELEASE_DZ = 0.04, 0.06
PHASES = ["reach", "descend", "close", "lift", "carry", "lower", "open", "retreat"]
os.makedirs("data/diag", exist_ok=True)

kp = np.load("data/perception/keyposes.npz")
suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(TASK_ID)
init_states = suite.get_task_init_states(TASK_ID)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)


def quintic(p0, p1, sec):
    sec = max(sec, 1.875 * np.linalg.norm(p1 - p0) / V_MAX)
    n = max(int(round(sec * HZ)), 2)
    tau = np.linspace(0, 1, n + 1)[1:]
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return p0 + s[:, None] * (p1 - p0)


def plan_for(take, obs):
    key = lambda f: kp[f"{take}_{f}"]
    J, B = obs["orange_juice_1_pos"], obs["basket_1_pos"]
    home = obs["robot0_eef_pos"].copy()
    g, r, rest = key("grasp")[:2], key("release")[:2], key("rest")[:2]
    vh, vs = r - g, B[:2] - J[:2]
    ang = np.arctan2(vs[1], vs[0]) - np.arctan2(vh[1], vh[0])
    scale = np.linalg.norm(vs) / np.linalg.norm(vh)
    Rz = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    approach = (J[:2] + scale * Rz @ (rest - g)) - J[:2]
    approach /= np.linalg.norm(approach) + 1e-9
    lift = np.clip(key("lift") - key("grasp")[2], args.lift_min, args.lift_max)
    reach_h = max(key("t_grasp") - 0.4, 0.8)
    carry_h = key("t_release") - key("t_grasp")
    slow = min(2.0, (MAX_STEPS - 45) / HZ / (reach_h + carry_h))
    reach_s, carry_s = reach_h * slow, carry_h * slow
    zg = J[2] + GRASP_DZ
    pre = np.r_[J[:2] + 0.03 * approach, zg + 0.10]
    grasp, up = np.r_[J[:2], zg], np.r_[J[:2], zg + lift]
    above, release = np.r_[B[:2], zg + lift], np.r_[B[:2], zg + RELEASE_DZ]
    plan = [(quintic(home, pre, 0.6 * reach_s), OPEN), (quintic(pre, grasp, 0.4 * reach_s), OPEN),
            (np.repeat(grasp[None], 15, 0), CLOSE), (quintic(grasp, up, 0.25 * carry_s), CLOSE),
            (quintic(up, above, 0.50 * carry_s), CLOSE), (quintic(above, release, 0.25 * carry_s), CLOSE),
            (np.repeat(release[None], 10, 0), OPEN), (quintic(release, release + [0, 0, 0.10], 1.0), OPEN)]
    ref = np.concatenate([p for p, _ in plan])
    grip = np.concatenate([np.full(len(p), gr) for p, gr in plan])
    phase = np.concatenate([np.full(len(p), i) for i, (p, _) in enumerate(plan)])
    info = dict(lift=lift, reach_s=reach_s, carry_s=carry_s,
                approach_deg=np.degrees(np.arctan2(approach[1], approach[0])),
                map_rot_deg=np.degrees(ang), map_scale=scale,
                human_dist_cm=100 * np.linalg.norm(vh))
    return ref, grip, phase, home, info


def run(take, layout):
    env.reset()
    obs = env.set_init_state(init_states[layout])
    for _ in range(10):
        obs, *_ = env.step(np.r_[np.zeros(6), OPEN])
    others = [k for k in obs if k.endswith("_pos") and "_to_" not in k and not k.startswith("robot0")
              and k not in ("orange_juice_1_pos", "basket_1_pos")]
    others0 = {k: obs[k].copy() for k in others}
    ref, grip, phase, home, info = plan_for(take, obs)
    q_home = obs["robot0_eef_quat"].copy()
    B0 = obs["basket_1_pos"].copy()
    tr = {k: [] for k in ["eef", "juice", "basket", "width", "tilt"]}
    frames, prev, success_step = [], home, None
    for i, (p, gr, ph) in enumerate(zip(ref, grip, phase)):
        eef = obs["robot0_eef_pos"]
        a_pos = np.clip((p - prev) / STEP_M + 0.6 * (p - eef) / STEP_M, -1, 1)
        rot_err = (Rot.from_quat(q_home) * Rot.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec()
        a_rot = np.clip(0.5 * rot_err / 0.1, -1, 1)
        obs, _, done, _ = env.step(np.r_[a_pos, a_rot, gr])
        prev = p
        if done and success_step is None:
            success_step = i
        w = obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]
        tilt = np.degrees(np.linalg.norm(
            (Rot.from_quat(q_home) * Rot.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec()))
        j, b = obs["orange_juice_1_pos"], obs["basket_1_pos"]
        tr["eef"].append(obs["robot0_eef_pos"].copy()); tr["juice"].append(j.copy())
        tr["basket"].append(b.copy()); tr["width"].append(w); tr["tilt"].append(tilt)
        moved = 100 * np.linalg.norm(b[:2] - B0[:2])
        im = np.ascontiguousarray(cv2.resize(obs["agentview_image"][::-1, ::-1], (512, 512)))
        cv2.putText(im, f"{i:3d} {PHASES[ph]}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(im, f"grip {100 * w:.1f}cm  juice z {100 * j[2]:.1f}cm  tilt {tilt:.0f}deg",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(im, f"basket moved {moved:.1f}cm", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 80, 80) if moved > 1 else (255, 255, 0), 2)
        frames.append(im)
    tr = {k: np.array(v) for k, v in tr.items()}
    tr.update(phase=phase, B0=B0, success=success_step is not None or bool(env.check_success()),
              others={k[:-4]: np.linalg.norm(obs[k] - others0[k]) for k in others})
    return tr, frames, info


def classify(tr):
    j, e, ph, w, b = tr["juice"], tr["eef"], tr["phase"], tr["width"], tr["basket"]
    reasons = []
    pushed = np.linalg.norm(j[ph < 2, :2] - j[0, :2], axis=1).max()
    if pushed > 0.01:
        reasons.append(f"pushed carton {100 * pushed:.1f} cm while approaching")
    w_closed = w[np.flatnonzero(ph == 2)[-1]]
    lifted = (j[:, 2] - j[0, 2]) > 0.03
    if not lifted.any():
        reasons.append("missed: closed on air" if w_closed < 0.01
                       else f"grasped (opening {100 * w_closed:.1f} cm) but never lifted")
    else:
        carry = (ph >= 3) & (ph <= 5)
        gap = np.linalg.norm(e - j, axis=1)
        k0 = np.flatnonzero(carry)[0]
        grew = np.flatnonzero(carry & (gap - gap[k0] > 0.04))
        if len(grew):
            reasons.append(f"dropped during {PHASES[ph[grew[0]]]} at step {grew[0]}")
    moved = np.linalg.norm(b[:, :2] - tr["B0"][:2], axis=1)
    if moved.max() > 0.01:
        k = np.flatnonzero(moved > 0.01)[0]
        reasons.append(f"hit basket during {PHASES[ph[k]]} at step {k}, "
                       f"carton z {100 * j[k, 2]:.1f} cm, basket moved {100 * moved.max():.0f} cm")
    d_end = np.linalg.norm(j[-1, :2] - b[-1, :2])
    if d_end > 0.08:
        reasons.append(f"carton ended {100 * d_end:.0f} cm from basket centre")
    for name, d in tr["others"].items():
        if d > 0.02:
            reasons.append(f"knocked {name} {100 * d:.0f} cm")
    if tr["tilt"].max() > 20:
        reasons.append(f"gripper tilted up to {tr['tilt'].max():.0f} deg")
    return reasons or ["no problem detected"]


def plot(tr, name):
    t = np.arange(len(tr["width"])) / HZ
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t, 100 * tr["eef"][:, 2], label="gripper z")
    ax[0].plot(t, 100 * tr["juice"][:, 2], label="carton z")
    ax[0].set_ylabel("cm"); ax[0].legend()
    ax[1].plot(t, 100 * tr["width"]); ax[1].set_ylabel("gripper opening cm")
    ax[2].plot(t, 100 * np.linalg.norm(tr["eef"] - tr["juice"], axis=1), label="gripper to carton")
    ax[2].plot(t, 100 * np.linalg.norm(tr["juice"][:, :2] - tr["B0"][:2], axis=1),
               label="carton to basket start (xy)")
    ax[2].plot(t, 100 * np.linalg.norm(tr["basket"][:, :2] - tr["B0"][:2], axis=1), label="basket moved")
    ax[2].set_ylabel("cm"); ax[2].set_xlabel("s"); ax[2].legend()
    for bnd in np.flatnonzero(np.diff(tr["phase"])):
        for a in ax:
            a.axvline((bnd + 1) / HZ, c="0.8", lw=0.8)
        ax[0].text((bnd + 1) / HZ, ax[0].get_ylim()[1], PHASES[tr["phase"][bnd + 1]], fontsize=7, va="top")
    fig.suptitle(name); fig.tight_layout(); fig.savefig(f"data/diag/{name}.png", dpi=100); plt.close(fig)


if args.failed:
    rows = list(csv.DictReader(open("data/replay_log.csv")))
    clean = [r for r in rows if r["noisy"] == "0"]
    jobs = [(r["take"], int(r["layout"])) for r in clean if r["success"] == "0"]
    jobs += [(r["take"], int(r["layout"])) for r in clean if r["success"] == "1"][:args.compare]
else:
    jobs = [(args.take, args.layout)]

summary = []
for take, layout in jobs:
    tr, frames, info = run(take, layout)
    name = f"{take}_L{layout}{args.tag}"
    imageio.mimsave(f"data/diag/{name}.mp4", frames, fps=HZ)
    plot(tr, name)
    reasons = classify(tr)
    verdict = "SUCCESS" if tr["success"] else "FAIL"
    carry_z = 100 * tr["juice"][:, 2].max()
    summary.append(dict(name=name, verdict=verdict, carry_z_cm=round(carry_z, 1),
                        **{k: round(float(v), 2) for k, v in info.items()}, reasons="; ".join(reasons)))
    print(f"{name:18} {verdict:7} lift {100 * info['lift']:4.1f}cm carton max z {carry_z:5.1f}cm "
          f"steps {len(tr['width'])} | {'; '.join(reasons)}")

with open(f"data/diag/summary{args.tag}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
print(f"\nvideos, plots and summary{args.tag}.csv in data/diag/")
env.close()