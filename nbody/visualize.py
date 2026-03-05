#!/usr/bin/env python3
"""
visualize.py  —  Animate the galaxy N-body simulation output.

Usage:
    python3 visualize.py simulation.bin           # live animation
    python3 visualize.py simulation.bin out.gif   # save to GIF

Requires: numpy, matplotlib
    pip install numpy matplotlib
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ─── Load binary simulation file ─────────────────────────────────────────────
def load(path: str):
    """
    Binary format written by GalaxySimulation::run():
        Header : int n_bodies, int n_snapshots   (8 bytes)
        Frames : for each snapshot:
                   n_bodies × (float x, float y, float z)  (12 bytes/body)

    Returns
    -------
    n_bodies  : int
    positions : ndarray of shape (n_snapshots, n_bodies, 3), dtype float32
    """
    with open(path, "rb") as f:
        n_bodies, n_snap = struct.unpack("ii", f.read(8))
        raw = np.frombuffer(f.read(n_bodies * n_snap * 3 * 4), dtype="f4")

    positions = raw.reshape(n_snap, n_bodies, 3)
    print(f"Loaded '{path}': {n_bodies} bodies × {n_snap} snapshots")
    return n_bodies, positions


# ─── Build and run the animation ─────────────────────────────────────────────
def animate(path: str, save_to: str = None):
    n_bodies, snaps = load(path)
    n_snap = snaps.shape[0]

    # Determine plot bounds from the 97th-percentile star distance
    # (skip body 0 = black hole which stays near origin).
    star_xy = snaps[:, 1:, :2]                                 # (snap, star, xy)
    star_r  = np.sqrt((star_xy ** 2).sum(axis=-1))             # (snap, star)
    lim     = float(np.percentile(star_r, 97)) * 1.4           # a little margin

    # ── Set up figure ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 9), facecolor="#050512")
    ax.set_facecolor("#050512")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title("Galaxy N-Body Simulation  (top-down view)", color="white", pad=10)
    ax.set_xlabel("x  [sim units]", color="#666688")
    ax.set_ylabel("y  [sim units]", color="#666688")
    ax.tick_params(colors="#444466")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    # Stars: small, semi-transparent blue-white dots.
    stars = ax.scatter(
        [], [], s=0.5, c="#aaccff", alpha=0.55,
        linewidths=0, rasterized=True)

    # Central black hole: bright gold star marker.
    bh = ax.scatter(
        [], [], s=130, c="#ffe050", marker="*", zorder=10,
        edgecolors="#ff9900", linewidths=0.6, label="Sgr A* (Black Hole)")

    # Info text (top-left corner).
    info = ax.text(
        0.02, 0.97, "", transform=ax.transAxes,
        color="white", fontsize=9, va="top", family="monospace")

    ax.legend(facecolor="#0d0d22", edgecolor="#333355",
              labelcolor="white", fontsize=9, loc="upper right")

    # ── Animation update function ─────────────────────────────────────────
    def update(frame):
        pos = snaps[frame]                       # (n_bodies, 3)
        stars.set_offsets(pos[1:, :2])           # all stars except BH
        bh.set_offsets(pos[0:1, :2])             # body 0 = black hole
        info.set_text(
            f"Snapshot : {frame + 1:3d} / {n_snap}\n"
            f"Bodies   : {n_bodies}")
        return stars, bh, info

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_snap,
        interval=60,      # ms between frames
        blit=True,
        repeat=True)

    # ── Save or display ───────────────────────────────────────────────────
    if save_to:
        print(f"Saving animation to '{save_to}' (this may take a while)…")
        ani.save(save_to, writer="pillow", fps=16, dpi=110)
        print("Saved.")
    else:
        plt.tight_layout()
        plt.show()


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    animate(input_file, output_file)
