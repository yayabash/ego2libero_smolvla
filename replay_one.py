"""Replay one human take in LIBERO: keyposes + timing -> robot reference -> closed-loop tracking."""
import argparse
import os

import imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--take", default="take_01")
ap.add_argument("--layout", type=int, default=0)
ap.add_argument("--grasp_dz", type=float, default=0.04, help="gripper height above juice origin")
ap.add_argument("--release_dz", type=float, default=0.06, help="release height above grasp height")
ap.add_argument("--slow", type=float, default=2.0, help="time stretch for human timing")
ap.add_argument("--rot_gain", type=float, default=0.5, help="0 disables orientation hold")
ap.add_argument("--video", action="store_true")
args = ap.parse_args()

HZ, STEP_M, TASK_ID = 20, 0.0098, 9                     # STEP_M measured by the probe
OPEN, CLOSE = -1.0, 1.0
kp = np.load("data/perception/keyposes.npz")
key = lambda f: kp[f"{args.take}_{f}"]

# --- simulator ---------------------------------------------------------------
suite = benchmark.get_benchmark_dict()["libero_object"]()
task = suite.get_task(TASK_ID)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(TASK_ID)[args.layout])
for _ in range(10):
    obs, *_ = env.step(np.r_[np.zeros(6), OPEN])

J, B = obs["orange_juice_1_pos"].copy(), obs["basket_1_pos"].copy()
home, q_home = obs["robot0_eef_pos"].copy(), obs["robot0_eef_quat"].copy()

# --- human -> sim mapping: grasp lands on the juice, release on the basket ----
g, r, rest = key("grasp")[:2], key("release")[:2], key("rest")[:2]
vh, vs = r - g, B[:2] - J[:2]
ang = np.arctan2(vs[1], vs[0]) - np.arctan2(vh[1], vh[0])
scale = np.linalg.norm(vs) / np.linalg.norm(vh)
Rz = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
to_sim = lambda p: J[:2] + scale * Rz @ (p - g)
approach = to_sim(rest) - J[:2]
approach /= np.linalg.norm(approach) + 1e-9

lift = np.clip(key("lift") - key("grasp")[2], 0.10, 0.20)
reach_s = max(key("t_grasp") - 0.4, 0.8) * args.slow
carry_s = (key("t_release") - key("t_grasp")) * args.slow

z_grasp = J[2] + args.grasp_dz
pre = np.r_[J[:2] + 0.03 * approach, z_grasp + 0.10]
grasp = np.r_[J[:2], z_grasp]
up = np.r_[J[:2], z_grasp + lift]
above_basket = np.r_[B[:2], z_grasp + lift]
release = np.r_[B[:2], z_grasp + args.release_dz]

# --- reference trajectory: quintic segments + gripper holds -----------------
def quintic(p0, p1, sec):
    n = max(int(round(sec * HZ)), 2)
    tau = np.linspace(0, 1, n + 1)[1:]
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return p0 + s[:, None] * (p1 - p0)

plan = [
    (quintic(home, pre, 0.6 * reach_s), OPEN),
    (quintic(pre, grasp, 0.4 * reach_s), OPEN),
    (np.repeat(grasp[None], 15, 0), CLOSE),               # let the gripper close
    (quintic(grasp, up, 0.25 * carry_s), CLOSE),
    (quintic(up, above_basket, 0.50 * carry_s), CLOSE),
    (quintic(above_basket, release, 0.25 * carry_s), CLOSE),
    (np.repeat(release[None], 10, 0), OPEN),              # let go
    (quintic(release, release + [0, 0, 0.10], 1.0), OPEN),
]
ref = np.concatenate([p for p, _ in plan])
grip = np.concatenate([np.full(len(p), gr) for p, gr in plan])

# --- closed-loop tracking: feedforward (planned step) + feedback (error) ----
frames, errors, prev, success = [], [], home.copy(), False
for p, gr in zip(ref, grip):
    eef = obs["robot0_eef_pos"]
    errors.append(np.linalg.norm(p - eef))
    a_pos = np.clip((p - prev) / STEP_M + 0.6 * (p - eef) / STEP_M, -1, 1)
    rot_err = (Rot.from_quat(q_home) * Rot.from_quat(obs["robot0_eef_quat"]).inv()).as_rotvec()
    a_rot = np.clip(args.rot_gain * rot_err / 0.1, -1, 1)
    obs, _, done, _ = env.step(np.r_[a_pos, a_rot, gr])
    prev = p
    if args.video:
        frames.append(obs["agentview_image"][::-1, ::-1])
    success = success or bool(done)
success = success or bool(env.check_success())

print(f"{args.take} layout {args.layout} grasp_dz {args.grasp_dz:+.2f}: "
      f"{'SUCCESS' if success else 'fail'} | {len(ref)} steps ({len(ref) / HZ:.1f} s) | "
      f"tracking error mean {100 * np.mean(errors):.1f} cm, max {100 * np.max(errors):.1f} cm | "
      f"lift {100 * lift:.0f} cm | map rotation {np.degrees(ang):.0f} deg, scale {scale:.2f} | "
      f"juice now {np.round(obs['orange_juice_1_pos'], 3)}")
if args.video:
    out = f"data/replay_{args.take}_L{args.layout}_dz{args.grasp_dz:+.2f}.mp4"
    imageio.mimsave(out, frames, fps=HZ)
    print("saved", out)
env.close()