# pylint: disable=g-bad-file-header
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Simple matplotlib rendering of a rollout prediction against ground truth.

Usage (from parent directory):

`python -m learning_to_simulate.render_rollout --rollout_path={OUTPUT_PATH}/rollout_test_1.pkl`

Where {OUTPUT_PATH} is the output path passed to `train.py` in "eval_rollout"
mode.

It may require installing Tkinter with `sudo apt-get install python3.7-tk`.

"""

import pickle
import argparse
import os

from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

TYPE_TO_COLOR = {
    3: "black",   # Boundary particles.
    0: "green",   # Rigid solids.
    7: "magenta", # Goop.
    6: "gold",    # Sand.
    5: "blue",    # Water.
}


def main():
    if not args.rollout_path:
        raise ValueError("A `rollout_path` must be passed.")
    with open(args.rollout_path, "rb") as file:
        rollout_data = pickle.load(file)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    plot_info = []
    for ax_i, (label, rollout_field) in enumerate(
        [("Ground truth", "ground_truth_rollout"),
         ("Prediction", "predicted_rollout")]):
        # Append the initial positions to get the full trajectory.
        trajectory = np.concatenate(
            [rollout_data["initial_positions"],
             rollout_data[rollout_field]],
            axis=0
        )
        ax = axes[ax_i]
        ax.set_title(label)
        bounds = rollout_data["metadata"]["bounds"]
        ax.set_xlim(bounds[0][0], bounds[0][1])
        ax.set_ylim(bounds[1][0], bounds[1][1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(1.0)
        points = {
            particle_type: ax.plot([], [], "o", ms=2, color=color)[0]
            for particle_type, color in TYPE_TO_COLOR.items()
        }
        plot_info.append((ax, trajectory, points))

    # Unpack trajectories for convenience
    # plot_info[0] -> Ground truth, plot_info[1] -> Prediction
    gt_trajectory = plot_info[0][1]
    pred_trajectory = plot_info[1][1]

    num_steps = gt_trajectory.shape[0]

    def update(step_i):
        outputs = []
        for _, trajectory, points in plot_info:
            for particle_type, line in points.items():
                mask = rollout_data["particle_types"] == particle_type
                line.set_data(
                    trajectory[step_i, mask, 0],
                    trajectory[step_i, mask, 1],
                )
                outputs.append(line)
        return outputs

    # Frames used for both PDF saving and GIF animation
    frame_indices = np.arange(0, num_steps, args.step_stride)

    # ---------- NEW: save each view as separate PDFs per frame ----------
    base_dir = os.path.dirname(args.rollout_path)

    rollout_frames_dir = os.path.join(base_dir, f"rf={args.rollout_path[-5]}",  "frames_rollout_pdf")
    gt_frames_dir = os.path.join(base_dir, f"rf={args.rollout_path[-5]}", "frames_ground_truth_pdf")
    os.makedirs(rollout_frames_dir, exist_ok=True)
    os.makedirs(gt_frames_dir, exist_ok=True)

    bounds = rollout_data["metadata"]["bounds"]
    particle_types = rollout_data["particle_types"]

    def save_single_view_pdf(step_i, trajectory, title, out_path):
        fig_single, ax_single = plt.subplots(figsize=(5, 5))
        ax_single.set_title(title)
        ax_single.set_xlim(bounds[0][0], bounds[0][1])
        ax_single.set_ylim(bounds[1][0], bounds[1][1])
        ax_single.set_xticks([])
        ax_single.set_yticks([])
        ax_single.set_aspect(1.0)
        for particle_type, color in TYPE_TO_COLOR.items():
            mask = particle_types == particle_type
            ax_single.plot(
                trajectory[step_i, mask, 0],
                trajectory[step_i, mask, 1],
                "o",
                ms=2,
                color=color,
            )
        fig_single.savefig(out_path, bbox_inches="tight")
        plt.close(fig_single)

    for i, step_i in enumerate(frame_indices):
        # Update main figure (for GIF)
        update(step_i)

        # Save rollout (prediction) only
        rollout_pdf_path = os.path.join(
            rollout_frames_dir, f"frame_{i:05d}_rollout.pdf"
        )
        save_single_view_pdf(
            step_i, pred_trajectory, "Prediction", rollout_pdf_path
        )

        # Save ground truth only
        gt_pdf_path = os.path.join(
            gt_frames_dir, f"frame_{i:05d}_ground_truth.pdf"
        )
        save_single_view_pdf(
            step_i, gt_trajectory, "Ground truth", gt_pdf_path
        )

    print(f"Saved rollout PDFs to {rollout_frames_dir}")
    print(f"Saved ground-truth PDFs to {gt_frames_dir}")
    # -------------------------------------------------------------------

    generated_animation = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        interval=10,
    )
    writer = animation.PillowWriter(fps=24)
    out_path = os.path.join(base_dir, "rollout.gif")
    generated_animation.save(out_path, writer=writer)
    print(f"Saved animation to {out_path}")
    plt.show(block=args.block_on_show)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout_path", type=str, required=True,
        help="Path to rollout pickle file"
    )
    parser.add_argument(
        "--step_stride", type=int, default=3,
        help="Stride of steps to skip."
    )
    parser.add_argument(
        "--block_on_show", type=bool, default=True,
        help="For test purposes."
    )

    args = parser.parse_args()
    main()
