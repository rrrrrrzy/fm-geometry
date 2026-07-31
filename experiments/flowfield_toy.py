#!/usr/bin/env python
"""Flow-field figure: the geometry of FM uncertainty on a learned toy flow field.

Trains ONE conditional flow-matching velocity net ``v_theta(x, s, o)``
(``fmaccel.toy``) on a ring of
observations, each carrying a Gaussian-mixture ACTION target: some **unimodal / tight**
(a *certain* field), some **multimodal** (an *aleatoric* fork), plus held-out **OOD**
observations sitting in the gaps between training angles (an *epistemic* extrapolation).

The figure realizes the paper's geometric claim:

  * a **certain** field is an affine isotropic contraction toward a single endpoint — its
    denoising trajectory is near-straight (low ``accel``) and the resampled posterior is
    tight (low spread);
  * an **uncertain** field (multimodal, or off the training support) departs from that
    affine template — the trajectory bends (high ``accel``) and the resampled posterior
    spreads.

``accel`` here is computed with the **exact same estimator** used on π₀.₅/LIBERO
(``fmaccel.geometry.accel.whole_chunk_curvature``), so the toy number is the same quantity as
the real-model result. The resampled-noise posterior spread (fixed obs, N resampled noise
seeds run to the clean action) is the toy stand-in for the Monte-Carlo resample GT
(``mean_std``, narrative §1.2).

We pick ONE representative observation per scenario (unimodal / multimodal / OOD), draw its
learned flow field as a progress-colored streamplot with the sampled denoising trajectories
overlaid, and **highlight one noise→action trajectory** per panel; the three panels are
concatenated horizontally. Every panel is annotated with its free ``accel`` and its
posterior spread so the accel↔spread ordering is legible at a glance.

Run (canonical env; ~1 min on GPU for the default 2-D field)::

    python experiments/flowfield_toy.py
    python experiments/flowfield_toy.py \
        --steps 40000 --seed 0 --out outputs/flowfield/toy

Outputs (into ``--out``): ``flowfield.png/.pdf`` (the 3-panel figure),
``all_fields.png`` (every trained + OOD observation, for picking/debugging), and
``summary.json`` (per-panel accel + spread + fidelity + the chosen observation indices).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # scripts/ -> for _bootstrap
import _bootstrap  # noqa: F401,E402  (repo root on sys.path + .env)


# ----------------------------------------------------------------- palette (dataviz)
# Validated reference palette (light surface); roles, not raw hex, are used below.
# --- palette: experiments/palette (paper-wide tone; lightness/saturation
#     tuned per role, but the four base hues are kept so every main-exp figure matches).
NAVY = "#364F6B"        # color_plan[0] — flow field lines
TEAL = "#3FC1C9"        # color_plan[1] — (reserved for other main-exp figures)
NEUTRAL = "#F5F5F5"     # color_plan[2] — chart surface
PINK = "#FC5185"        # color_plan[3] — target-distribution cloud + highlighted trajectory
NAVY_DARK = "#243546"   # darkened navy — the trajectory's noise-start dot
RED = "#e60000"         # pure red — target mode means (×)

INK = "#0b0b0b"         # text — black
INK_MUTED = "#52514e"   # secondary text (ticks / axis labels) — dark gray
SURFACE = NEUTRAL
FRAME = "#0b0b0b"        # panel border — black
GRID = "#0b0b0b"        # annotation-box edge — black
FIELD_LINE = NAVY       # (retained for reference — streamlines now use TAU_CMAP)
HIGHLIGHT = PINK        # (retained for markers — trajectory now uses TAU_CMAP)
MODE_X = RED            # target mode means (× markers) — pure red
ENDPOINT_CLOUD = PINK   # the target-distribution point cloud (sampled endpoints)
NOISE_PT = NAVY_DARK    # the trajectory's noise start

# --- denoising-progress colormap (the "trajectory-bundle" colour channel).
# Encodes τ ∈ [0, 1]: 0 = pure noise (start of the denoise integration), 1 = clean
# action (endpoint). Anchors are all mid- to fully-saturated so the streamplot reads
# with weight on a light surface — but the τ=0 end is still lighter than τ=1 to give
# a monotone luminance sweep the reader can follow along a single trajectory.
# The sweep is deep-steel-blue → teal → plum → deep-pink; the noise-start dot uses
# NAVY_DARK as a separate marker (dot ≠ streamline) so it remains identifiable.
def _make_tau_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "tau_progress",
        ["#2C5F87",   # τ=0 — deep steel blue (readable, weighted, not navy-black)
         "#1F8E96",   # τ≈0.33 — darker TEAL (paper palette shifted -L)
         "#8E3E76",   # τ≈0.66 — deep plum (interpolant between teal and pink)
         "#C81E5F"],  # τ=1 — deep PINK (paper palette shifted -L; matches endpoint hue)
        N=256)


TAU_CMAP = _make_tau_cmap()
TAU_LABEL = r"denoising progress $\tau$ (noise $\to$ action)"

# Times New Roman isn't installed in the the project environment; STIXGeneral is the bundled, metric-
# compatible Times substitute (matches Times New Roman). Prefer the real face if present.
SERIF_STACK = ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
FONT_RC = {"font.family": "serif", "font.serif": SERIF_STACK, "mathtext.fontset": "stix",
           "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE}


# ============================================================== scenario selection
def _nearest_mode_dist(endpoints, means):
    """Mean over samples of the distance from each endpoint to its nearest target mode."""
    import numpy as np
    d = np.linalg.norm(endpoints[:, None, :] - means[None, :, :], axis=-1)  # (N, K)
    return float(d.min(axis=1).mean())


def _pick_unimodal(records):
    """Best-learned tight unimodal obs (lowest posterior spread among the confident ones)."""
    uni = [r for r in records if (not r["is_ood"]) and r["n_modes"] == 1]
    if not uni:
        raise ValueError("no unimodal training observation found — raise n_obs or lower n_multimodal")
    return min(uni, key=lambda r: r["spread_mean_std"])


def _pick_multimodal(records):
    """Clearest multimodal fork: prefer a well-covered 2-mode target, else the highest-spread one."""
    multi = [r for r in records if (not r["is_ood"]) and r["n_modes"] > 1]
    if not multi:
        raise ValueError("no multimodal training observation found — raise n_multimodal")
    two = [r for r in multi if r["n_modes"] == 2]
    pool = two or multi
    # highest posterior spread = the most visibly-forking field
    return max(pool, key=lambda r: r["spread_mean_std"])


def _pick_ood(records, avoid=None):
    """OOD observation to feature — the one whose extrapolated field bends the *most*
    (highest accel = most uncertain-looking), so it reads visually distinct from the
    multimodal panel's two-cluster fork. ``avoid`` (a set of mode-mean arrays) lets us
    skip an OOD whose endpoint cloud looks too much like the featured multimodal one.
    """
    ood = [r for r in records if r["is_ood"]]
    if not ood:
        raise ValueError("no OOD observation found — set n_ood >= 1")
    return max(ood, key=lambda r: r["accel_median"])


# ============================================================== per-observation record
def build_record(net, task, cfg, noise, pooled_std, action_dim, num_steps, *, seed_offset):
    """Integrate one observation's learned field and read its geometry.

    Draws ``noise`` -> action trajectories on the field sliced at ``task.o``, then reads
    (a) the 2-D flow-field arrays for the streamplot, (b) the free per-trajectory ``accel``
    with the SHARED ``pooled_std`` (so the three panels' accels are on one scale), and
    (c) the resampled-noise posterior spread (the toy resample-GT).
    """
    import numpy as np
    import torch

    from fmaccel.geometry.accel import whole_chunk_curvature
    from fmaccel.toy.model import euler_sample, obs_velocity_fn

    dev, dt = cfg.resolved_device(), cfg.dtype
    vfn = obs_velocity_fn(net, task.o)
    _, traj, s_grid = euler_sample(vfn, noise.shape[0], action_dim, num_steps=num_steps,
                                   device=dev, dtype=dt, noise=noise.to(dev, dt), return_traj=True)
    traj = traj.detach().cpu().numpy()                    # (T+1, N, d)
    s_grid = s_grid.detach().cpu().numpy()                # (T+1,)

    # velocities along the rollout (pre-step), for the binned flow field
    with torch.no_grad():
        vel = torch.stack([vfn(torch.as_tensor(traj[i]).to(dev, dt), float(s_grid[i]))
                           for i in range(num_steps)]).detach().cpu().numpy()  # (T, N, d)

    # free accel via the PRODUCTION estimator: each trajectory is a chunk_size=1 "chunk",
    # x_t shaped (N, T+1, 1, d); the shared pooled_std puts every panel on one scale.
    x_t = np.transpose(traj, (1, 0, 2))[:, :, None, :].astype(np.float32)      # (N, T+1, 1, d)
    accel = whole_chunk_curvature(x_t, action_dim, fixed_std=pooled_std)["accel"]  # (N,)

    endpoints = traj[-1]                                   # (N, d)
    noise_pts = traj[0]                                    # (N, d)
    spread_mean_std = float(endpoints.std(axis=0).mean())  # toy resample-GT (mean_std)
    # mean pairwise endpoint distance (the chunk-divergence-mean analog), for cross-check
    from itertools import combinations
    if endpoints.shape[0] <= 256:
        pair = np.array([np.linalg.norm(endpoints[i] - endpoints[j])
                         for i, j in combinations(range(endpoints.shape[0]), 2)])
    else:                                                  # subsample pairs for speed
        idx = np.random.default_rng(0).integers(0, endpoints.shape[0], (4000, 2))
        pair = np.linalg.norm(endpoints[idx[:, 0]] - endpoints[idx[:, 1]], axis=-1)
    spread_pair_mean = float(pair.mean())

    rec = {
        "name": task.name, "is_ood": bool(task.is_ood), "n_modes": int(task.n_modes),
        "accel": accel, "accel_mean": float(accel.mean()), "accel_median": float(np.median(accel)),
        "spread_mean_std": spread_mean_std, "spread_pair_mean": spread_pair_mean,
        # arrays for plotting
        "pos": traj[:num_steps], "vel": vel, "progress": s_grid[:num_steps],
        "noise_pts": noise_pts, "endpoints": endpoints, "traj": traj,
        "means": (task.target.means.detach().cpu().numpy() if task.target is not None else None),
    }
    if task.target is not None:
        rec["endpoint_mode_dist"] = _nearest_mode_dist(endpoints, rec["means"])
    return rec


def highlight_index(rec):
    """Index of the trajectory to draw boldly: a *representative-accel* sample whose
    noise→action chord is long enough to read on the panel.

    We take a band of trajectories around the median accel (so the highlighted path is
    typical of the panel, not cherry-picked), then within that band prefer a long
    noise→endpoint displacement — a short chord (noise happening to start near the mode)
    is visually uninformative. For a target-bearing obs we additionally require the
    endpoint to land near a real mode (a clean successful sample).
    """
    import numpy as np
    accel = rec["accel"]
    order = np.argsort(accel)
    lo = max(0, len(order) // 2 - 25)
    band = order[lo: lo + 50]                              # ~median-accel trajectories
    chord = np.linalg.norm(rec["endpoints"][band] - rec["noise_pts"][band], axis=-1)
    if rec["means"] is None:                               # OOD: longest chord among the band
        return int(band[int(np.argmax(chord))])
    # trained obs: among the band, keep those landing near a mode, then take the longest chord
    d = np.linalg.norm(rec["endpoints"][band][:, None, :] - rec["means"][None, :, :], axis=-1).min(axis=1)
    near = d <= (np.median(d) + 1e-9)                      # the mode-hitting half of the band
    cand = np.where(near)[0]
    if cand.size == 0:
        cand = np.arange(len(band))
    return int(band[cand[int(np.argmax(chord[cand]))]])


# ============================================================================ plotting
def _bin_field(pos, vel, progress, *, grid, min_count):
    """Bin (pos, vel, progress) 2-D rollout onto a regular grid mean-field for streamplot.

    pos/vel: (T, N, 2); progress: (T,). Empty / sparse cells -> NaN (streamplot skips them).
    """
    import numpy as np
    P = pos.reshape(-1, 2)
    W = vel.reshape(-1, 2)
    Tp = np.broadcast_to(progress[:, None], pos.shape[:2]).reshape(-1)
    xmin, xmax = float(P[:, 0].min()), float(P[:, 0].max())
    ymin, ymax = float(P[:, 1].min()), float(P[:, 1].max())
    pad = 0.04 * max(xmax - xmin, ymax - ymin, 1e-9)
    xe = np.linspace(xmin - pad, xmax + pad, grid + 1)
    ye = np.linspace(ymin - pad, ymax + pad, grid + 1)
    ix = np.clip(np.searchsorted(xe, P[:, 0], side="right") - 1, 0, grid - 1)
    iy = np.clip(np.searchsorted(ye, P[:, 1], side="right") - 1, 0, grid - 1)
    flat = iy * grid + ix
    n = grid * grid
    cnt = np.zeros(n, np.int64)
    U = np.zeros(n); V = np.zeros(n); Tm = np.zeros(n)
    np.add.at(cnt, flat, 1)
    np.add.at(U, flat, W[:, 0]); np.add.at(V, flat, W[:, 1]); np.add.at(Tm, flat, Tp)
    ok = cnt > 0
    U[ok] /= cnt[ok]; V[ok] /= cnt[ok]; Tm[ok] /= cnt[ok]
    mask = cnt.reshape(grid, grid) < min_count
    nan = lambda a: np.where(mask, np.nan, a.reshape(grid, grid))
    return {"cx": 0.5 * (xe[:-1] + xe[1:]), "cy": 0.5 * (ye[:-1] + ye[1:]),
            "U": nan(U), "V": nan(V), "Tm": nan(Tm), "extent": [xe[0], xe[-1], ye[0], ye[-1]]}


def _shared_limits(recs, pad=0.06):
    """One square (xlim, ylim) covering every featured panel's noise + endpoint clouds.

    A shared window aligns the equal-aspect panel boxes (so titles line up) AND makes the
    posterior spread directly comparable by eye across panels — a tight certain cloud reads
    as small next to a diffuse uncertain one on the very same axes.
    """
    import numpy as np
    xs = np.concatenate([np.concatenate([r["noise_pts"][:, 0], r["endpoints"][:, 0]]) for r in recs])
    ys = np.concatenate([np.concatenate([r["noise_pts"][:, 1], r["endpoints"][:, 1]]) for r in recs])
    lo = np.array([np.percentile(xs, 0.5), np.percentile(ys, 0.5)])
    hi = np.array([np.percentile(xs, 99.5), np.percentile(ys, 99.5)])
    ctr = 0.5 * (lo + hi)
    half = 0.5 * float((hi - lo).max()) * (1.0 + pad)     # square: max half-range + padding
    return (float(ctr[0] - half), float(ctr[0] + half)), (float(ctr[1] - half), float(ctr[1] + half))


def _panel(ax, rec, kind_label, cfg, *, grid, min_count, n_faint, xlim=None, ylim=None,
           show_noise_cloud=False):
    """One flow-field panel: streamplot + faint trajectory cloud + one bold highlighted path.

    ``show_noise_cloud`` toggles the diffuse noise-sample scatter (the s=0 starts); the paper
    figure hides it so only the target (endpoint) cloud remains as the visible posterior.
    """
    import numpy as np

    import numpy as _np
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    # Path progress τ (0 = noise start, 1 = action end), derived from step index so the cmap
    # reads consistently across the toy (s = 0 → 1) and real (times = 1 → 0) conventions —
    # both scripts feed rec["pos"] step-major, so step-index-normalized is always "noise→action".
    T_pos = rec["pos"].shape[0]
    path_tau = _np.linspace(0.0, 1.0, T_pos, dtype=_np.float32) if T_pos > 1 else _np.zeros(1, _np.float32)
    b = _bin_field(rec["pos"], rec["vel"], path_tau, grid=grid, min_count=min_count)
    # streamlines shaded by mean binned progress τ ∈ [0, 1] — the "trajectory bundle" reading:
    # each streamline segment is coloured by how far into the denoise integration it lives, so
    # the reader sees noise (dark) → action (pink) *in the field itself*, without a slice-of-τ
    # commitment. NaN cells (empty bins) are handled by streamplot's masking machinery.
    tau_norm = Normalize(vmin=0.0, vmax=1.0)
    strm = ax.streamplot(b["cx"], b["cy"], b["U"], b["V"], color=b["Tm"],
                         cmap=TAU_CMAP, norm=tau_norm,
                         density=1.25, linewidth=0.9, arrowsize=0.9, zorder=1)
    # (the faint trajectory-cloud lines were removed — they visually competed with the
    # τ-shaded streamplot; ``n_faint`` is retained in the signature for backward compat
    # but now controls nothing.)
    _ = n_faint
    traj = rec["traj"]
    # noise + endpoint clouds (noise cloud hidden by default — target cloud is the posterior)
    if show_noise_cloud:
        ax.scatter(rec["noise_pts"][:, 0], rec["noise_pts"][:, 1], s=3, c=INK, alpha=0.10, zorder=2)
    ax.scatter(rec["endpoints"][:, 0], rec["endpoints"][:, 1], s=5, c=ENDPOINT_CLOUD, alpha=0.16,
               edgecolors="none", zorder=2)
    # target mode means (x); OOD has none
    if rec["means"] is not None:
        ax.scatter(rec["means"][:, 0], rec["means"][:, 1], marker="x", c=MODE_X, s=95,
                   linewidths=2.4, zorder=5, label="target mode")
    # the highlighted noise→action trajectory — drawn as a τ-shaded LineCollection (a proper
    # "bundle" line: colour marches from noise-start (dark) to action-endpoint (pink) along
    # the path, giving the eye an explicit read-out of denoise progress). We use the same
    # TAU_CMAP as the streamlines so the whole panel speaks a single colour language.
    hi = highlight_index(rec)
    p = rec["traj"][:, hi, :]
    tau = _np.linspace(0.0, 1.0, p.shape[0], dtype=_np.float32)     # τ=0 at noise start
    segs = _np.stack([p[:-1], p[1:]], axis=1)                       # (T, 2, 2) point pairs
    tau_seg = 0.5 * (tau[:-1] + tau[1:])                            # per-segment mid-τ
    lc = LineCollection(segs, cmap=TAU_CMAP, norm=None, linewidths=2.6,
                        capstyle="round", zorder=6, alpha=0.98)
    lc.set_array(tau_seg)
    lc.set_clim(0.0, 1.0)
    lc.set_label("sampled trajectory")
    ax.add_collection(lc)
    ax.scatter([p[0, 0]], [p[0, 1]], s=70, c=NOISE_PT, edgecolors="white", linewidths=1.2,
               zorder=7, label="noise start")
    ax.scatter([p[-1, 0]], [p[-1, 1]], marker="*", s=240, c=PINK, edgecolors="white",
               linewidths=1.2, zorder=7, label="action (endpoint)")

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(colors=INK_MUTED, labelsize=8)
    for sp in ax.spines.values():                          # darker navy panel frame
        sp.set_edgecolor(FRAME)
        sp.set_linewidth(1.6)
    ax.set_xlabel("action dim 0", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("action dim 1", fontsize=9, color=INK_MUTED)

    # title + annotation: the resample-GT posterior spread (TV(v) omitted from the plot)
    ax.set_title(kind_label, fontsize=12.5, color=INK, pad=8, fontweight="bold")
    txt = f"posterior spread = {rec['spread_mean_std']:.2f}"
    ax.text(0.035, 0.965, txt, transform=ax.transAxes, va="top", ha="left", fontsize=11,
            color=INK, bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=FRAME, alpha=0.95))
    return strm


def _make_scalar_mappable():
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=TAU_CMAP)
    sm.set_array([])
    return sm


def _style_tau_cbar(cbar, *, orientation="horizontal", label=None):
    cbar.set_label(label or TAU_LABEL, fontsize=9, color=INK_MUTED)
    cbar.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(FRAME)
    cbar.outline.set_linewidth(0.8)
    if orientation == "horizontal":
        cbar.ax.set_xticklabels(["noise ($\\tau{=}0$)", "$\\tau{=}0.5$", "action ($\\tau{=}1$)"])
    else:
        cbar.ax.set_yticklabels(["noise", "$0.5$", "action"])


def _add_tau_colorbar(fig, axes, *, orientation="horizontal", label=None,
                      pad=0.02, shrink=0.55, aspect=32):
    """Attach a shared τ (denoise progress) colorbar tied to TAU_CMAP.

    Kept small and neutral — the coloured streamlines / trajectory carry the reading; the
    colorbar just anchors the τ=0 (noise) → τ=1 (action) semantic in a single place. Uses
    matplotlib's automatic layout (steals space from ``axes``); for exact placement in a
    reserved slot use :func:`_add_tau_colorbar_to_ax`.
    """
    cbar = fig.colorbar(_make_scalar_mappable(), ax=axes, orientation=orientation, pad=pad,
                        shrink=shrink, aspect=aspect, ticks=[0.0, 0.5, 1.0])
    _style_tau_cbar(cbar, orientation=orientation, label=label)
    return cbar


def _add_tau_colorbar_to_ax(fig, cax, *, orientation="horizontal", label=None):
    """Draw the τ colorbar into a pre-reserved axes ``cax`` (GridSpec slot).

    Unlike :func:`_add_tau_colorbar`, this doesn't steal space — the caller has already
    allocated the slot — so a legend can be placed in a neighbouring slot without matplotlib
    re-flowing the layout under ``bbox_inches="tight"``.
    """
    cbar = fig.colorbar(_make_scalar_mappable(), cax=cax, orientation=orientation,
                        ticks=[0.0, 0.5, 1.0])
    _style_tau_cbar(cbar, orientation=orientation, label=label)
    return cbar


def make_figure(uni, multi, ood, cfg, out_dir, *, grid, min_count, n_faint):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.5))
        labels = ["Certain — unimodal target", "Uncertain — multimodal (aleatoric)",
                  "Uncertain — OOD (epistemic)"]
        xlim, ylim = _shared_limits((uni, multi, ood))     # one square window -> aligned + comparable
        for ax, rec, lab in zip(axes, (uni, multi, ood), labels):
            _panel(ax, rec, lab, cfg, grid=grid, min_count=min_count, n_faint=n_faint,
                   xlim=xlim, ylim=ylim)

        fig.tight_layout(rect=(0, 0.08, 0.955, 1))         # strip for legend at bottom + cbar at right
        # shared τ colorbar on the RIGHT — steals a thin vertical strip so the streamlines
        # stay visually dominant; the cbar just anchors the noise → action semantic.
        _add_tau_colorbar(fig, list(axes), orientation="vertical",
                          pad=0.02, shrink=0.85, aspect=28)
        # shared legend (single row, bottom) — identity is never color-alone
        handles, lbls = axes[1].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=len(lbls), frameon=False,
                   fontsize=10.5, bbox_to_anchor=(0.5, -0.02))
        for ext in ("png", "pdf"):                          # transparent bg (drops the surface fill)
            fig.savefig(out_dir / f"flowfield.{ext}", dpi=200,
                        bbox_inches="tight", transparent=True)
        plt.close(fig)


def make_individual_panels(uni, multi, ood, cfg, out_dir, *, grid, min_count, n_faint):
    """Emit each of the three featured panels as its own standalone figure.

    Same content / shared square window / per-panel legend as ``make_figure`` — just split
    into one file per scenario so the left / middle / right panels can be used independently.
    Files: ``flowfield_{left,middle,right}.png/.pdf``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Certain — unimodal target", "Uncertain — multimodal (aleatoric)",
              "Uncertain — OOD (epistemic)"]
    slots = ["left", "middle", "right"]
    xlim, ylim = _shared_limits((uni, multi, ood))         # same window as the combined figure

    with plt.rc_context(FONT_RC):
        for rec, lab, slot in zip((uni, multi, ood), labels, slots):
            # Explicit 2x2 layout: [main | τ-cbar (right)] on top, [legend spanning both]
            # on the bottom. Reserving explicit slots keeps bbox_inches="tight" from
            # re-flowing the composition into overlaps.
            fig = plt.figure(figsize=(6.2, 6.2))
            gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 0.045], height_ratios=[1.0, 0.09],
                                  wspace=0.05, hspace=0.24)
            ax = fig.add_subplot(gs[0, 0])
            cax = fig.add_subplot(gs[0, 1])
            lax = fig.add_subplot(gs[1, :])
            _panel(ax, rec, lab, cfg, grid=grid, min_count=min_count, n_faint=n_faint,
                   xlim=xlim, ylim=ylim)
            _add_tau_colorbar_to_ax(fig, cax, orientation="vertical")
            lax.axis("off")
            handles, lbls = ax.get_legend_handles_labels()
            lax.legend(handles, lbls, loc="center", ncol=len(lbls), frameon=False, fontsize=9,
                       borderaxespad=0.0)
            for ext in ("png", "pdf"):
                fig.savefig(out_dir / f"flowfield_{slot}.{ext}", dpi=200,
                            bbox_inches="tight", transparent=True)
            plt.close(fig)


def make_overview(records, cfg, out_dir, *, grid, min_count, n_faint):
    """Every trained + OOD observation, for picking/debugging the representative panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(records)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    with plt.rc_context(FONT_RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.2 * nrow), squeeze=False)
        flat = [axes[i // ncol][i % ncol] for i in range(nrow * ncol)]
        for ax, rec in zip(flat, records):
            tag = "OOD" if rec["is_ood"] else f"K={rec['n_modes']}"
            _panel(ax, rec, f"{rec['name']}  ({tag})", cfg, grid=grid, min_count=min_count, n_faint=n_faint)
        for ax in flat[n:]:
            ax.axis("off")
        fig.suptitle("all observations (trained + OOD) — TV(v) vs posterior spread", fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / "all_fields.png", dpi=130, bbox_inches="tight", transparent=True)
        plt.close(fig)


# ================================================================================ config
def build_config(args):
    """A ToyConfig tuned for a clean 2-D flow-field figure (several uni/multi/OOD obs)."""
    from fmaccel.toy.config import ToyConfig
    cfg = ToyConfig(
        action_dim=2,            # 2-D so the flow field is directly plottable
        obs_dim=8,
        obs_layout="circle",
        n_obs=args.n_obs,        # ring of observations
        n_multimodal=args.n_multimodal,
        n_ood=args.n_ood,
        modes_range=(2, 2),      # 2-mode forks read cleanest as a visual fork
        mode_sep=2.0,
        mode_box=2.3,
        sigma_range=(0.06, 0.13),  # tight modes -> a genuinely *certain* unimodal field
        weight_temp=0.3,
        samples_per_obs=4096,
        obs_jitter=0.01,
        steps=args.steps,
        num_steps=args.num_steps,
        seed=args.seed,
        device=args.device or "cuda",
        output_dir=args.out,
    )
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="outputs/flowfield/toy", help="output directory")
    p.add_argument("--steps", type=int, default=40000, help="training steps (2-D field learns fast)")
    p.add_argument("--num-steps", type=int, default=40, help="Euler denoise steps T (raise for smoother geometry)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda / cpu (default cuda, falls back to cpu)")
    p.add_argument("--n-obs", type=int, default=6, help="observations on the ring")
    p.add_argument("--n-multimodal", type=int, default=3, help="how many of them are multimodal")
    p.add_argument("--n-ood", type=int, default=3, help="held-out OOD gap-midpoint observations")
    p.add_argument("--n-traj", type=int, default=512, help="resampled-noise trajectories per observation")
    p.add_argument("--grid", type=int, default=26, help="streamplot bin resolution")
    p.add_argument("--min-count", type=int, default=5, help="drop flow-field cells with fewer pairs")
    p.add_argument("--n-faint", type=int, default=40, help="faint trajectory cloud lines per panel")
    # explicit overrides for the featured obs (default: auto by the selectors above)
    p.add_argument("--uni", default=None, help="force the unimodal panel to this obs name (e.g. o3)")
    p.add_argument("--multi", default=None, help="force the multimodal panel obs name")
    p.add_argument("--ood", default=None, help="force the OOD panel obs name (e.g. ood1)")
    return p.parse_args()


def _by_name(records, name):
    for r in records:
        if r["name"] == name:
            return r
    raise ValueError(f"observation {name!r} not found; available: {[r['name'] for r in records]}")


def main():
    import sys
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    import numpy as np
    import torch

    args = parse_args()
    cfg = build_config(args)
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("config:\n" + json.dumps(dataclasses.asdict(cfg), indent=2, default=str))
    print(f"device -> {cfg.resolved_device()}")

    from fmaccel.toy import data, model
    from fmaccel.toy.analysis import fidelity

    # 1. data + train the LIVE conditional flow net
    train_tasks = data.build_train_tasks(cfg)
    ood_tasks = data.build_ood_tasks(cfg)
    print(f"\n{len(train_tasks)} train obs "
          f"({sum(t.n_modes > 1 for t in train_tasks)} multimodal) + {len(ood_tasks)} OOD")
    print("\ntraining v_theta(x, s, o):")
    net, _ = model.train_obs_flow(train_tasks, cfg)

    # 2. one shared noise batch (reused across every panel) + one shared per-dim accel scale
    dev, dt = cfg.resolved_device(), cfg.dtype
    g = torch.Generator().manual_seed(cfg.seed + 4242)
    noise = torch.randn(args.n_traj, cfg.action_dim, generator=g, dtype=dt)

    # pooled per-dim std across ALL observations' trajectories -> shared accel z-score scale
    all_tasks = train_tasks + ood_tasks
    from fmaccel.toy.model import euler_sample, obs_velocity_fn
    endpts_stack = []
    traj_all = []
    for t in all_tasks:
        _, traj, _ = euler_sample(obs_velocity_fn(net, t.o), args.n_traj, cfg.action_dim,
                                  num_steps=cfg.num_steps, device=dev, dtype=dt,
                                  noise=noise.to(dev, dt), return_traj=True)
        traj_all.append(np.transpose(traj.detach().cpu().numpy(), (1, 0, 2)))  # (N, T+1, d)
    pooled = np.concatenate(traj_all, axis=0)                                   # (n_obs*N, T+1, d)
    pooled_std = pooled.reshape(-1, cfg.action_dim).std(axis=0).astype(np.float32)  # (d,)
    print(f"\nshared accel per-dim std (pooled over all obs): {np.array2string(pooled_std, precision=3)}")

    # 3. per-observation records (geometry + accel + posterior spread + fidelity)
    print("\nper-observation geometry (TV(v) = free denoise-path curvature; spread = resample-GT):")
    records = []
    for k, t in enumerate(all_tasks):
        rec = build_record(net, t, cfg, noise, pooled_std, cfg.action_dim, cfg.num_steps,
                           seed_offset=k)
        if t.target is not None:
            rec["fidelity"] = fidelity(net, t, cfg)
        records.append(rec)
        tag = "OOD " if rec["is_ood"] else f"K={rec['n_modes']} "
        fid = rec.get("fidelity", {})
        print(f"  {rec['name']:6s} {tag:5s} TV(v)(med)={rec['accel_median']:5.2f} "
              f"spread={rec['spread_mean_std']:.3f} pair={rec['spread_pair_mean']:.3f} "
              f"{'ed=%.2f learned=%s' % (fid.get('energy_dist_norm', float('nan')), fid.get('learned')) if fid else ''}")

    # 4. pick the representative obs per scenario (auto, or forced via flags)
    uni = _by_name(records, args.uni) if args.uni else _pick_unimodal(records)
    multi = _by_name(records, args.multi) if args.multi else _pick_multimodal(records)
    ood = _by_name(records, args.ood) if args.ood else _pick_ood(records)
    print(f"\nfeatured panels:  unimodal={uni['name']}  multimodal={multi['name']}  OOD={ood['name']}")
    print(f"  TV(v)(med):   uni={uni['accel_median']:.2f} < multi={multi['accel_median']:.2f} , ood={ood['accel_median']:.2f}")
    print(f"  spread:       uni={uni['spread_mean_std']:.3f} < multi={multi['spread_mean_std']:.3f} , ood={ood['spread_mean_std']:.3f}")

    # 5. figures — combined 3-panel, per-panel standalones, and the debug overview
    make_figure(uni, multi, ood, cfg, out_dir, grid=args.grid, min_count=args.min_count,
                n_faint=args.n_faint)
    make_individual_panels(uni, multi, ood, cfg, out_dir, grid=args.grid,
                           min_count=args.min_count, n_faint=args.n_faint)
    make_overview(records, cfg, out_dir, grid=args.grid, min_count=args.min_count, n_faint=args.n_faint)

    # 6. numeric record
    def _slim(r):
        return {k: r[k] for k in ("name", "is_ood", "n_modes", "accel_mean", "accel_median",
                                  "spread_mean_std", "spread_pair_mean") if k in r} | \
               ({"energy_dist_norm": r["fidelity"]["energy_dist_norm"],
                 "learned": r["fidelity"]["learned"]} if "fidelity" in r else {})
    summary = {
        "config": dataclasses.asdict(cfg),
        "featured": {"unimodal": uni["name"], "multimodal": multi["name"], "ood": ood["name"]},
        "pooled_accel_std": [float(x) for x in pooled_std],
        "observations": [_slim(r) for r in records],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote: {out_dir}/flowfield.png/.pdf, "
          f"flowfield_{{left,middle,right}}.png/.pdf, all_fields.png, summary.json")


if __name__ == "__main__":
    main()
