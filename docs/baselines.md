# Baselines — implementation reference for `fmaccel/detectors/`

> **What this is.** A per-detector audit. For **every** detector — the free `accel` under test and
> each literature baseline — this records the **source** (paper + official repo), the **faithful
> algorithm** as published *and* as implemented in the official code, **how this repo implements
> it**, the **defaults**, and the **fidelity status + caveats** to know before reporting a number.
>
> Comparing against baselines is only meaningful if they are implemented in good faith, so the
> guiding rule is: stay consistent with the original paper *and* its open-source code, and document
> every place we deviate. Each `detectors/*.py` docstring names its own source; this file is the
> consolidated account.
>
> **Audit provenance.** The fidelity claims below were produced by reading each official repo
> (citations are to actual files/functions) and adversarially re-checked. Legend:
>
> | mark | meaning |
> |---|---|
> | ✅ **Faithful** | matches the paper **and** the official repo up to behaviour-preserving, documented choices |
> | 🔶 **Adapted (flagged)** | a *necessary, documented* deviation (e.g. diffusion→flow-matching, per-chunk vs windowed), internally correct |
> | ⚠️ **Divergence (unflagged)** | differs from paper/repo and is **not** documented in the code — should be fixed or at least documented |
> | 🐛 **Bug** | a correctness error on the **as-shipped default path** (the path the harness actually runs) |
>
> **Fix status:** every *confirmed* divergence below has been **fixed** in the code and re-verified
> (import purity without lerobot, per-detector in-distribution vs out-of-distribution behaviour).
> Entries keep the "what the official method is / what was wrong" account for the record, each
> followed by a **✓ Fixed** line, so a reader can see what was corrected rather than only the
> corrected state. The `accel` numerics were not touched by any of it.

---

## 0. The contract every detector shares

Read [`base.py`](base.py) first. Three pieces make the cross-detector comparison fair and the code uniform:

- **`ChunkRecord`** — a numpy bundle describing **one re-plan decision** (`k=1` observational
  sampling). Always carries `x_t (T+1, chunk, act)` (the recorded denoise path) + `action_dim` +
  bookkeeping (`env_step / task_id / task_group / n_exec / time`). Optional signals are populated on
  demand and stay `None` otherwise: `v_t`, `chunk_actions`, `obs_emb`, `hidden`, `resample
  (N, exec, act)`, `prev_chunk`, `prev_resample (N, exec, act)` (the previous decision's set),
  `resample_full` / `prev_resample_full (N, chunk, act)`, `extra`.
  ⚠️ `resample`/`prev_resample` are the **executed-window** slice (`first_actions == n_exec`, the
  window `accel` is labelled on). A **cross-time** detector must use the `_full` pair instead: two
  consecutive *executed* windows cover disjoint absolute timesteps, so differencing them measures
  motion, not plan change. This bit silently neutered `stac` until 2026-07-27 — see its entry below.
- **`Detector`** — sets `name`, `requires: frozenset ⊆ REQUIREABLE`, `online: bool`, `supervised:
  bool`, and implements `fit(...)` (no-op default) + `score(rec) -> float` (+ optional
  `score_stream(recs)` for a history-dependent probe like SAFE-LSTM). The harness guarantees every
  signal in `requires` is present before calling `score`. **`supervised=True`** (only `safe`) flips
  the fit contract from success-only `fit(success_rollouts)` to failure-labeled
  `fit([(records, success), …])`; the harness routes and holds out the two fit families separately.
- **Score convention (load-bearing).** `score(rec)` returns a scalar where **lower = more confident /
  less likely to fail** (the native polarity of `accel`). Every literature score here is natively
  *higher = OOD/uncertain/inconsistent = failure*, which already agrees with "higher value ⇒ flag",
  so they are returned **as-is** — no detector needs a sign flip, and none applies one wrongly. The
  harness (`failure_detection_eval`) treats higher = failure uniformly (Mann-Whitney AUROC,
  `tpr_at_fpr` alarms on the high side); there is **no per-detector polarity special-casing**.

**Two scoring paths, same `score` call** (see `docs/baselines.md §2`):

| path | who runs it | where | builds |
|---|---|---|---|
| **online** (`requires ⊆ {x_t, v_t}`) | `accel`, `straightness`, `sparc` | the policy server's predict path | `ChunkRecord` from `sample_chunks_with_traj` |
| **post-hoc** (everything else) | resample (`ace`/`stac`/`oracle`) / embedding / loss / `safe` | [`pipelines/detector_score.py`](../pipelines/detector_score.py) over `fm/rollouts/*.npz` (+ `chunk_divergence/*.npz`, `obs_emb/`, `hidden_states/`) | `ChunkRecord` per chunk (`prev_resample` threaded forward); runs the shared harness |

> **What lives in the harness, not the detector.** Every literature method's published *alarm* adds a
> sliding-window aggregation + a success-calibrated conformal/quantile threshold (FIPER window-SUM,
> FAIL-Detect functional CP band, Diff-DAgger 0.99-quantile + K-window).
> Our detectors deliberately return the **raw per-chunk scalar**; windowing/calibration are the shared
> harness's job so the cross-detector comparison stays threshold-free (raw-stream AUROC). When you
> compare against a paper's headline TPR/FPR, remember you are comparing the *score*, not their full
> banded detector.

---

## 1. Free / online detectors (read the recorded denoise path; no training, no GPU)

### `accel` — the reference method under test ✅ (project-internal)
- **What.** `accel = Σ_t‖Δv_{t+1}−Δv_t‖ / ⟨‖Δv_t‖⟩` of the per-dim z-scored denoise path — the free
  temporal-bend proxy validated against the MC-resample posterior (ρ≈+0.86 on π₀.₅/LIBERO).
- **Impl.** [`accel.py`](accel.py) is a thin wrapper over `geometry/accel.score_chunks`
  → `chunk_geometry.whole_chunk_curvature/whole_chunk_prefix_accel`, so the detector signal is the
  same numerics the server records as `chunk_accels`. Default `mode="accel_prefix:7"` = executed-window
  accel read at denoise depth 9. Polarity native (lower = straighter = confident), no flip.
- **Not a literature baseline** — it is the project's own proxy; fidelity-to-paper is N/A.
- ⚠️ **Caveat (load-bearing): default `fixed_std=None` self-normalizes each chunk.** `score` feeds one
  chunk as a `(1, T+1, chunk, act)` batch, so with no `fixed_std` the per-dim z-score is taken over
  *that one path*. The validated detector numbers come from **offline per-episode-std** labels;
  the per-chunk self-norm reranks chunks (our own normalization ablation measured
  corr ≈ 0.30) and **will not reproduce them**. To match the paper scale, pass the demo-distribution
  `fixed_std` (`action_dim`/window matched). The post-hoc
  `detector_score` path builds detectors with bare defaults, so it currently runs at the self-norm
  scale — see §6.
- Minor: with an `*exec*`/`*prefix*` mode and `n_exec=None`, the bare `geometry/accel.score_chunks`
  would silently fall back to the whole chunk (a different, undocumented signal). `AccelDetector` now
  **guards this** — it raises `ValueError` rather than scoring the wrong window (`accel.py:63-68`). The
  standard harness sets `n_exec = min(n_action_steps, chunk_size)`, so the guard only trips on ad-hoc use
  that forgets to pass `n_exec`.

### `straightness` — saturating geometric sibling of accel ✅ (project-internal)
- **What.** `straightness = ‖x_T−x_0‖ / Σ_t‖Δx_t‖` of the per-dim z-scored flattened chunk path
  (1 = straight). Returns `1 − straightness` (accel convention: higher = curvier = failure).
- **Impl.** [`straightness.py`](straightness.py) → `chunk_geometry.whole_chunk_straightness`. Standard
  chord/arc ratio; a straight path returns exactly 1.0.
- **Caveat.** Under plain CFM, straightness **saturates** (~97% of chunks in [0.99, 1.0]) — expected to
  *underperform* accel and reported as the near-equivalent saturating comparator, not a novel method.
- ⚠️ Same online single-chunk self-normalization as `accel` (N=1 ⇒ std over that chunk's denoise
  points only), **and** there is no `fixed_std` path here (`whole_chunk_straightness` takes no fixed
  std), so the online straightness is not directly comparable to any offline straightness labels.

### `sparc` — spectral arc length (movement smoothness) 🔶/⚠️
- **Source.** Balasubramanian et al. 2015, *On the analysis of movement smoothness* (J NeuroEng Rehabil
  12:112). Canonical reference implementation: **`siva82kb/SPARC` → `sparc()`**.
- **Faithful algorithm (reference `sparc()`):** `nfft = 2**(ceil(log2(N)) + padlevel)`; FFT magnitude
  spectrum; **normalize by its peak**; keep the band `f ≤ fc`; within it take the **contiguous index
  span** from the first to the last bin with normalized magnitude `≥ amp_th`; arc length
  `−Σ √((Δf/span)² + Δm²)` (more negative = less smooth). Defaults `padlevel=4`, `amp_th=0.05`,
  `fc=10 Hz` on a real sampling rate. Input is a **rest-to-rest scalar speed profile**.
- **Impl.** [`sparc.py`](sparc.py) `spectral_arc_length` reproduces the reference **bit-for-feature**
  (`padlevel=4`, `amp_th=0.05`, peak normalization, contiguous threshold span, span-normalized arc;
  `rfft` is equivalent to `fft` for a real signal). Returns `−SPARC` so higher = jerkier = failure.
- 🔶 **Adapted (flagged):** (a) computed on a **single ~16-step chunk** (cheap online), whereas the
  *streaming/windowed* form runs over the executed-action stream — a constant-velocity short chunk can
  leak and read as non-smooth (empirical AUROC ≈ 0.40 confirms the short-signal caveat); (b)
  multi-dim action → **per-dim-standardized Euclidean speed** `‖diff(a/std)‖`, vs the reference's
  scalar speed (this *reshapes*, not just rescales, the profile).
- ⚠️ **`fc=0.5` with `fs=1.0`** keeps the **full band up to Nyquist** (`fc/Nyquist = 1.0`), unlike the
  reference's band-limiting `fc/Nyquist ≈ 0.2`. Reasonable for a ~15-sample signal but under-documented.
- ⚠️ **Attribution:** the docstring's "used as a baseline in FAIL-Detect" was **incorrect** — the
  FAIL-Detect score family is logpZO / RND / CFM / PCA_kmeans / NatPN / DER; **SPARC is not one of
  them**. → **✓ Fixed:** docstring now cites only Balasubramanian (2015), and the `fc=0.5` full-band
  choice is documented.
- Minor: degenerate / `<3`-position chunks return `0.0` = the *most-confident* end (a masking default).

---

## 2. Resample-based detectors (read the K MC-resample candidates; post-hoc, numpy, no GPU)

> These read `rec.resample (N, chunk, act)`, built by `detector_score` from the
> `chunk_divergence/*.npz` candidates. ⚠️ Keep `N` and the binning/formula **distinct from the oracle's
> spread definition** so they don't silently reconstruct the GT.

### `ace` — FIPER's Action-Chunk Entropy leg 🔶/⚠️
- **Source.** FIPER (Römer et al., NeurIPS 2025, arXiv:2510.09459), repo `utiasDSL/fiper`
  (mirror `learnsyslab/fiper`), `entropy_eval.py`.
- **Faithful algorithm.** Per decision sample **B** action chunks `~π(·|O_t)` (B **task-specific**:
  32/32/256/30/256 across tasks — there is no single global default); bin **3D Cartesian
  end-effector position** (`required_actions:[position]`, columns 0,1,2) per horizon step; **base-2**
  Shannon entropy of the occupancy histogram (`p = counts/B`, biased plug-in MLE); aggregate over the
  horizon (**paper SUMs**, eq.4/10; **repo MEANs** = sum/len). Grid is **adaptive in limits** (per-
  decision min/max of the B endpoints + 1% buffer) but **fixed in cell size**, where the cell is
  precomputed once from calibration as `cell = cellsize_factor · R_d` (per-dim working-area range
  `R_d`; `cellsize_factor ≈ 0.03`).
- **Impl.** [`ace.py`](ace.py): occupancy histogram via `floor(points/cell)`, base-2 entropy, **MEAN
  over the horizon (repo-faithful)**, higher = more spread = failure (native). Sample source = the
  MC-resample endpoints (faithful in spirit — for a flow-matching policy, independent-noise resamples
  *are* i.i.d. draws from `π(·|O_t)`).
- ⚠️ **Cell was a fixed absolute scalar (`0.05`), never calibrated** — `fit()` was a no-op despite the
  docstring claiming FIPER-style calibration. → **✓ Fixed:** `fit(success_rollouts)` now calibrates a
  **per-dim cell `= cellsize_factor·R_d`** (`R_d` = per-dim working-area range over the success set,
  `cellsize_factor=0.03`, FIPER eq.11), and scoring uses **adaptive per-decision grid limits** (anchor
  at each decision's per-dim min). The scalar `cell` remains only as the no-calibration fallback.
- ⚠️ **`pos_dims=(0,1,2)` are joint-space, not Cartesian EE position** for this repo's policy
  (π₀.₅/LIBERO action dims are joint/delta-EE + gripper), so ACE measures first-3-dim occupancy,
  a different quantity than FIPER's working-area EE grid. → **Documented** (this is data-dependent — set
  `pos_dims` to the model's real position columns when known); not auto-fixable.
- 🔶 Horizon reduced to the **executed window** (`n_exec`) by default vs FIPER's full chunk `H` (flagged).
  → **✓ Fixed:** NaN resample rows are now masked before binning (were mapping to an int sentinel cell).
- **The standalone ACE leg is FIPER's *weaker* leg, not the full FIPER detector** (see `fiper`).

### `stac` — Statistical Temporal Action Consistency (Sentinel; a score in FAIL-Detect) ✅/🔶
- **Source.** Sentinel (Agia et al., CoRL 2024, arXiv:2410.04640), repo `agiachris/sentinel`
  (`bc/ood_detection/error_utils.py`); reused as a score in FAIL-Detect (`CXU-TRI/FAIL-Detect`
  `UQ_test/eval_load_baseline.py:STAC_UQ`). Audited verbatim against both repos (2026-07-01).
- **Faithful algorithm.** At decision `t`, draw a batch of `B` action chunks and, over the
  **overlapping future timesteps** of the previous vs current chunk, take the MMD (RBF, biased
  V-stat, median-heuristic bandwidth) between the two sample sets. Sentinel's overlap slice
  (`compute_temporal_error`): with exec horizon `k` and chunk length `h`, previous chunk's **TAIL**
  `prev[:, k:h]` vs current chunk's **HEAD** `curr[:, :h−k]` — the **same absolute future timesteps**.
  Binary gripper commands are **dropped** before the distance (`filter_gripper_action`); rotation is
  **kept** (the starred `STAC MMD*` default). MMD² = `mean(Kxx)+mean(Kyy)−2·mean(Kxy)` (**biased
  V-statistic**, diagonal included). Bandwidth `γ = 1/(2·median(pooled SQUARED pairwise distances,
  zeros excluded))` → kernel `exp(−‖·‖²/(2·median²))` (sklearn `rbf_kernel` has no `/2`, so `/2`
  lives in `γ`). Deployment = plain cumulative sum `η_t = Σ_{i≤t} MMD_i` (**NOT** CUSUM) + 95th-quantile
  **split-conformal** threshold calibrated on `M≈10–50` SUCCESS rollouts. Defaults `h=16`, `k=8/4`,
  `B=256/32`, `δ=0.05`.
- **Impl.** [`stac.py`](stac.py) `mmd2_rbf` reproduces the biased V-stat + Sentinel's median-of-squared
  γ **bit-for-form**; `StacDetector` reads `rec.resample_full` (current) + `rec.prev_resample_full`
  (previous, threaded forward by `detector_score`), slices the overlap (current head vs previous tail),
  drops `drop_dims` (the binary gripper column), flattens, returns the raw per-decision MMD² (the cumsum
  + conformal band is the shared harness's job, like every other baseline). Higher = inconsistent =
  failure (native, no flip). Synthetic-verified: a consistent re-plan (new chunk = old chunk shifted by
  `n_exec`) → MMD **exactly 0.0**; a shift injected on the overlap → 0.71.
- 🐛 **HIGH — FIXED 2026-07-27: the overlap window was truncated away upstream, so this detector never
  ran its own algorithm.** `chunk_divergence` writes `first_actions = n_action_steps` and
  `chunk_geometry._load_div_chunks` sliced `chunks[:, :, :first_actions]` — correct for accel/divergence
  (they want the *executed* plan's spread) but it left `rec.resample` with `chunk == n_exec` on **every**
  model. `overlap = chunk − n_exec` was therefore always 0 and the no-overlap fallback below always
  fired, comparing two **disjoint absolute time spans** (`t…t+k` vs `t−k…t`): a measure of "the robot
  moved", not "the plan changed". Fix: `requires` now names `resample_full`/`prev_resample_full` (the
  untruncated horizon); `_load_div_chunks(..., full_window=True)` returns it from the same single npz
  read, and the executed-window slice fed to `ace`/`oracle`/`fiper` is **bit-identical** to before.
  Effect at k=8, online CUSUM TPR@FPR=0.1: **restricting the MMD to the genuinely overlapping future
  window is what makes STAC work whenever `chunk > n_exec`** — every configuration with a real plan
  overlap moves from near-chance to strongly discriminative (`pi05:libero_all`, true overlap 40:
  0.20 → **0.93**). Configurations whose **native** `chunk == n_exec` have no overlap to recover and
  were unaffected (unchanged before/after).
- 🔶 **Adapted (flagged):** (a) `N` = the MC-resample count (the FD main-exp default is **k=4**; the
  saved candidate axis holds 32, so k is a free re-score) vs Sentinel's `B=256` — the biased V-stat has
  `O(1/N)` positive bias, so keep `N` FIXED across calibration/test (the median heuristic self-normalizes
  and the bias cancels in the conformal quantile); guard `N<2 → 0.0`. Measured k=4 vs k=8 on the
  overlap-fixed detector: within ±0.03 CUSUM TPR wherever a real overlap exists; the one noisier case is
  a native-no-overlap configuration, where the full-chunk fallback is noisiest. (b)
  **`chunk == n_exec`** (no plan overlap): Sentinel `assert h−k>0` has no in-paper
  fallback (its remedy is to reduce `k`); we fall back to the **full-chunk** MMD — this is what
  FAIL-Detect's `STAC_UQ` does (full-chunk flatten, no overlap slice), so it is **FAIL-Detect-style,
  NOT Sentinel-faithful**, flagged not silent.
- ⚠️ **Median convention (documented):** Sentinel uses median-of-**SQUARED** distances; FAIL-Detect's
  `median_trick_bandwidth` uses `2·(median of RAW distances)²` (`median(d²) ≠ (median d)²`). We follow
  **Sentinel** (median of squared) for the paper claim. First decision (no `prev_resample_full`) → `0.0`
  (matches FAIL-Detect's zero at `t=0`).
- ⚠️ **`drop_dims` is data-dependent:** default `None` keeps every action dim; pass the gripper
  column(s) for the policy (π₀.₅/LIBERO gripper is the LAST action dim) to match Sentinel's
  `filter_gripper_action`. History: `stac` was dropped in commit `315c9af` (with its `prev_resample`
  signal); restored 2026-07-01 and re-audited.

### `oracle_resample_spread` — the uncertainty GROUND TRUTH (oracle / upper bound, **NOT a baseline**) 🐛
- **What.** The MC-resample posterior spread (pairwise distance of the N resampled chunks) **is the
  project's uncertainty GT** — `accel` is *validated against* it. Drawn only as the upper-bound
  reference line; scoring it as a competitor double-counts the GT.
- **Impl.** [`oracle.py`](oracle.py) → `chunk_geometry.whole_chunk_divergence`.
- 🐛 **HIGH — the as-shipped default erased the GT magnitude.** With `scale_act=None` (the old default,
  and the path `detector_score` took) the per-dim scale was the resample cloud's **own** std *at this
  decision*. Both the pairwise distances and that std scale linearly with the cloud, so the score was
  **invariant to uniform scaling of the cloud** — a 100×-larger (= more uncertain) cloud scored
  identically, saturating in the high-uncertainty regime (verified: true spread 0.05/0.2/1.0/5.0 →
  self-std 8.66/9.24/9.60/9.58, non-monotonic). The `chunk_geometry` GT instead uses a **single
  run-pooled** `dim_scale` for every chunk, so magnitude is comparable — the basis of the ρ≈0.86 validation.
  - **✓ Fixed:** the per-chunk self-std branch is **removed**; the scale is now `scale_act` (constructor)
    → `rec.extra['dim_scale']` → **raw distance** (never self-std). `detector_score` surfaces the
    run-pooled `dim_scale` (previously loaded and discarded as `_scale`) into `rec.extra`, so the oracle
    equals the GT magnitude (verified: tight 0.16 vs 100× → 16.4, no longer collapsed). Does **not**
    affect the `chunk_geometry` validation or the headline `accel` AUROCs.
- Note: the oracle truncates to `n_exec` whereas `chunk_geometry` uses the `first_actions` window
  (`fa`) — a residual non-equality when they differ. Exclusion from competitive ranking is still
  enforced only by the verbose name + convention, not by harness/plot code (operator's responsibility).

---

## 3. Loss family (model-in-the-loop)

### `fm_loss` — Diff-DAgger's denoising loss, adapted to flow matching 🔶
- **Source.** Diff-DAgger (Lee et al., ICRA 2025, arXiv:2410.14868), repo `sean1295/DiffDAgger`.
- **Faithful idea.** Score a chunk by how well the policy denoises its **own** generated action across
  noise levels: at an in-distribution state the model reconstructs at every scale (low loss); OOD →
  loss spikes. Diff-DAgger MC-estimates the **DDPM ε-prediction loss** with **16×32 = 512** forwards,
  thresholds at the **0.99 quantile** of the demo-loss CDF + a **K-consecutive** window.
- **Impl.** [`fm_loss.py`](fm_loss.py) adapts the diffusion loss to the **flow-matching** policy: draw
  `(t, x0)`, form `x_t = (1−t)·x0 + t·â`, evaluate the policy velocity `v_θ(x_t, t, o)` and average
  `‖v_θ − (â − x0)‖²` (the CFM clean-target loss). Higher = worse self-reconstruction = failure
  (native). The 0.99-quantile/K-window *decision* is left to the harness.
- 🔶 **Adapted (flagged), and internally correct.** The diffusion→FM change is necessary (same model ⇒
  fair) and the in-file math is a **self-consistent rectified-flow clean-data parametrization**: the
  MSE is invariant to which endpoint is labelled `t=1`, and the velocity field the injected
  `rec.extra['fm_velocity']` callable must return is **uniquely determined by the coded interpolant +
  target** (lines 67–68). Adversarial check confirmed: given a correctly-built `fm_velocity`, the loss
  is ~0 on an in-distribution chunk.
- ⚠️ **Docstring wording is imprecise and could mislead the `fm_velocity` builder.** π₀.₅'s *native*
  convention is `x_t = t·noise + (1−t)·action`, `u = noise − action` (`modeling_pi05.py:733-735`), i.e.
  **the time-reverse + sign-flip** of this file's `x_t = (1−t)·x0 + t·â`, `target = â − x0`. So the
  injected callback must return `−v_θ(x_t, 1−t)` relative to π₀.₅'s raw velocity. The docstring calling
  this "π₀.₅'s CLEAN-target interpolant" / "match π₀.₅'s exact interpolant" reads as if the raw model
  velocity fits directly — it does **not**. **Add an in-distribution low-loss smoke gate** before
  trusting any numbers, and state the `−v_θ(x_t, 1−t)` contract explicitly.
- **Not free / GPU hand-off, synthetic-verified only.** Each decision needs many fresh re-noised
  velocity forwards (cannot reuse the recorded denoise path); the per-chunk `fm_velocity` is injected
  from a cached prefix / `ChunkResampleSession` (not yet wired). Defaults `m_t=16, m_noise=2` → **32**
  re-noisings (vs Diff-DAgger's 512), with a stratified midpoint t-grid vs uniform `t` (both
  reasonable, but document the reduced budget). RNG is seeded by `seed + 1000003·task_id + env_step`
  (`fm_loss.py:71`) so two chunks at the same `env_step` in different episodes/tasks don't draw identical
  noise, while staying fully reproducible.

---

## 4. Embedding-OOD density family (need the obs-embedding hook; `fit` on success-only)

> ⚠️ **Shared hand-off:** all of these read `rec.obs_emb`, a pooled prefix/conditioning vector the
> recorder does **not** capture yet (the model-specific obs-embedding hook is a GPU hand-off, see
> `baselines.md §4.1`). They are synthetic-verified only. **Use the same embedding at `fit` and
> `score`.** `fit` must see **closed-loop success** rollouts (not demos — demos are straighter and
> mis-calibrate the scale).

### `rnd_oe` — Random Network Distillation on the obs embedding ✅/⚠️
- **Source.** RND (Burda et al. 2018, arXiv:1810.12894); FIPER's OOD leg (`rnd_models.py`, the
  `RND_OE` variant); a top FAIL-Detect score.
- **Faithful algorithm.** A **frozen** random-init target `g` and a **trained** predictor `f_θ` map the
  policy **observation embedding** to a feature vector; train `f_θ` to regress `g` on **success-only**
  data; score `s = ‖f_θ(O) − g(O)‖₂` (FIPER/FAIL-Detect default `rnd_loss='l2'` = `PairwiseDistance(p=2)`
  — the **L2 norm**, not MSE). Higher = OOD = failure.
- **Impl.** [`rnd.py`](rnd.py): frozen target + trained predictor MLP; score = `((pred−target)²).sum().sqrt()`
  = **L2 norm**, which **matches** the FIPER/FAIL-Detect `'l2'` scoring. Polarity native. Deterministic
  seeding. (Training minimizes squared-L2 while scoring with the norm — fine; only a gradient-scale
  difference.)
- ⚠️ **Predictor and target were architecturally identical** (dropping RND's *deliberate* asymmetry) and
  the **obs embedding was z-scored** (FIPER's `RND_OE` uses `normalize_tensors.obs_embeddings=False`).
  → **✓ Fixed:** the predictor is now a deeper MLP than the target (verified 4 vs 2 Linear layers), both
  **orthogonal-init (gain √2, zero bias)** with a **frozen** target, and the obs embedding is fed **raw**
  (no `_mu/_sd`) — matching FIPER's recipe. Score remains the L2 norm (`= 'l2'`).
- Minor: `lr=1e-3` (FIPER 1e-4 cosine→1e-6), widths down-scaled (CPU MLP), `epochs=250`/`batch=256` match.
- *Don't conflate:* FAIL-Detect's `"RND"` is a `ConditionalUnet1D` on the **action chunk** with the obs
  embedding as `global_cond` (uses both `O_t, A_t`); ours follows **FIPER's** pure obs-embedding RND-OE.

### `logpzo` — flow-matching density of the obs embedding (FAIL-Detect's best score) 🐛
- **Source.** FAIL-Detect (Xu et al., RSS 2025, arXiv:2503.08558), repo `CXU-TRI/FAIL-Detect`
  (`UQ_baselines/logpZO/train.py`, `UQ_test/eval_load_baseline.py`).
- **Faithful algorithm (official code).** Train a CFM velocity net on **success-only** embeddings with
  the interpolant flowing **data → noise**: `x0 = O (data)`, `x1 = randn (noise)`, `vtrue = x1 − x0`,
  `x_t = (1−t)·x0 + t·x1`, loss `‖v − vtrue‖²`. **Score = one forward at `t=0`:** `z = O + v(O, 0)`,
  `score = ‖z‖²` = `.pow(2).sum(-1)` (**sum of squares**, not mean, not norm). Because data sits at
  `t=0`, the one Euler step maps ID embeddings near `N(0,I)` (small `‖z‖²`) and OOD far (large). It is
  a one-step Euler surrogate for `−log p(O)` (drops the CNF integration + the log-det Jacobian). **O_t
  only** (no action concat).
- **Impl.** [`logpzo.py`](logpzo.py): polarity (higher=OOD), the **sum-of-squares** score, success-only
  training, and the `O_t`-only choice all **match**.
- 🐛 **HIGH — orientation/test mismatch (confirmed against the repo).** The old `fit` trained the
  **opposite** orientation `noise → data` (`x0 = randn`, `x1 = data`), so at `t=0` the net only ever saw
  **noise** — but `score` applied the repo's `z = O + v(O, 0)`, feeding the **data** point `O` at the
  noise time. Empirically it still discriminated (not sign-inverted) but **substantially weaker** than
  the faithful map — undersells the headline FAIL-Detect baseline.
  - **✓ Fixed:** training now flows **data → noise** like the repo — `x0 = O (data, at t=0)`,
    `x1 = randn (noise)`, `target = x1 − x0`, `x_t = (1−t)·x0 + t·x1` — and keeps `z = O + v(O, 0)`. Now
    consistent (data at `t=0`). Re-verified: ID-vs-OOD AUROC 1.00 (mean-shift) / 0.99 (scale), up from
    the degraded pre-fix range. `std` floor raised to `1e-3`.
- 🔶 Architecture: 3-layer **MLP** with concatenated scalar time vs the repo's `ConditionalUnet1D`
  (`down_dims=[256,512,1024]`, `time_scale=100`) — a defensible from-scratch reimplementation, but a
  real capacity/inductive-bias difference (unflagged). Per-dim **standardization added** (repo scores on
  the raw padded embedding). `lr=1e-3`/AdamW/`batch=256` vs repo `1e-4`/Adam/`128` (`epochs=200` match).

### `pca_kmeans` / `knn` / `mahalanobis` — classical embedding density ✅/⚠️
- **Source.** FAIL-Detect's density family. **Only `PCA_kmeans` is in the official repo**
  (`UQ_baselines/PCA_kmeans/net_PCA.py`): `PCA(n_components = emb_dim)` (pure rotation, **no reduction**)
  → `KMeans(n_clusters=64)` on the **raw** embedding; score = Euclidean distance to the nearest centroid.
  `kNN` (Sun et al. 2022) and `Mahalanobis` (Lee et al. 2018) are **paper-level references, not released
  benchmark code.**
- **Impl.** [`density.py`](density.py), pure numpy: `pca_kmeans` (SVD + Lloyd k-means, nearest-centroid
  Euclidean), `knn` (k-th nearest in-distribution distance), `mahalanobis` (shrinkage covariance +
  `pinv`). Polarity native (distance higher = OOD), math sound. `mahalanobis` is affine-invariant so the
  pre-standardization is harmless.
- ⚠️ **Over-attribution:** **only `pca_kmeans` is a FAIL-Detect benchmark detector.** → **✓ Fixed:**
  module docstring + `baselines.md` now attribute `knn`/`mahalanobis` to Sun 2022 / Lee 2018 (paper-level,
  not FAIL-Detect repo).
- ⚠️ `pca_kmeans` **z-scored + reduced to 64 PCA dims**; `knn` used z-scoring. → **✓ Fixed:** the shared
  z-score is removed and each method uses its own faithful normalization — `pca_kmeans` on the **raw**
  embedding with **full-dim PCA** (`n_components=None`, pure rotation) + KMeans(64) + nearest-centroid
  Euclidean (matches the repo); `knn` **L2-unit-normalizes** (Sun 2022); `mahalanobis` on raw (affine-
  invariant). Re-verified: pca 0.999 / knn 0.876 / maha 1.000 ID-vs-OOD AUROC.

---

## 5. Assembled

### `fiper` — the published FIPER detector = AND(RND-OE, ACE) 🔶
- **Source.** FIPER (NeurIPS 2025), `utiasDSL/fiper`. The detector is the **conjunction** of two
  success-conformally-calibrated legs, combined by `np.minimum` of the **threshold-normalized** leg
  sequences (alarm = `min(normalized) > 1` ⇔ **both** legs above threshold), each leg first aggregated
  by a **sliding-window SUM** over `w` decisions and thresholded by a functional CP band.
- **Impl.** [`fiper.py`](fiper.py): `fit` trains RND-OE + sets each leg's threshold `τ` = the
  **α=0.95 quantile** of success scores; `score` returns `min(s_rnd/τ_rnd, s_ace/τ_ace)` — the correct
  **AND operator and polarity** (higher = both legs high = failure). Reporting AND (not a single leg) is
  the faithful FIPER.
- 🔶 **Adapted (flagged):** drops the window-SUM + functional CP band → a **single-α threshold-free
  min** for raw-stream AUROC (so the comparison stays α-independent). Now inherits the **fixed** `ace`
  (calibrated cell) and **fixed** `rnd` (asymmetric, no-norm) legs (see §2, §4). Needs both `obs_emb`
  (hand-off) and `resample`.
- ⚠️ `fit` did not calibrate ACE's cell, and `float(np.quantile(...)) or 1.0` substituted `1.0` only at
  an **exactly-0** success quantile (common for ACE), discontinuously rescaling that leg. → **✓ Fixed:**
  `fit` now calls `ace.fit(recs)` (calibrates the per-dim cell) and each leg threshold is floored to a
  **positive value tied to the leg's magnitude** (`max(s)` when the α-quantile is 0). Re-verified:
  ID-vs-OOD AUROC 0.997, ACE cell calibrated inside `fiper.fit`.

---

## 5b. Supervised probe (category-C control; needs FAILURE labels + the hidden hook)

### `safe` — SAFE's supervised failure probe on the action-expert last-hidden feature ✅/🔶
- **Source.** SAFE (Gu et al., NeurIPS 2025, arXiv:2506.09937, "Multitask Failure Detection for VLA
  Models"), repo `github.com/vla-safe/SAFE` (`failure_prob/`). Audited verbatim against `data/pizero.py`,
  `model/indep.py`, `model/lstm.py`, `utils/conformal/functional_predictor.py`, `conf/__init__.py`
  (2026-07-01).
- **What / why.** The paper's **category-C intrusive control** (uses FAILURE labels), the contrast that
  shows accel needs none. A *small* supervised probe reads the VLA's last-layer internal feature and
  emits a per-step failure score, on the premise that a VLA already "knows" about impending failure
  generically across tasks.
- **Faithful algorithm.** FEATURE (flow-matching/π₀ case = ours, `data/pizero.py`): the action-expert
  last-layer hidden tensor `(n_diff_steps, n_pred_horizon, d)` — for π₀.₅ the `suffix_out` fed
  to `action_out_proj` at each denoise step — reduced to `e_t ∈ R^d` by aggregating **horizon FIRST,
  then flow-step** via `process_tensor_idx_rel` (PizeroDatasetConfig default = **mean/mean**). **No PCA.**
  Two probes: **`mlp`** (`indep.py`, default, `cumsum=True`): `p_t = σ(g(e_t))`, SAFE score `s_t = Σ_{τ≤t}
  p_τ` (0<s_t<t, NOT a prob); trained so SUCCESS episodes minimize `Σ s_t` and FAILURE minimize `Σ(−s_t)`,
  inverse-class-freq + λ_reg L2. **`lstm`** (`lstm.py`): `s_t = σ(LSTM(e_{0:t})) ∈ [0,1]`, full-history
  (`n_history_steps=−1`), BCE against the trajectory label broadcast to every step (FAILURE = positive =
  `1−success`). Detection = **functional conformal prediction** (`functional_predictor.py`, Diquigiovanni
  2021): calibrate `μ_t` + band `δ_t = μ_t + q_{1−α}·h_t` on SUCCESS rollouts, flag `s_t > δ_t`; rollout
  AUROC uses running-max. Defaults: adam lr 1e-3 (lstm 3e-4), 1000 ep, batch 512, wd 1e-2, λ_reg 1.0, MLP
  2 layers/hidden 256, LSTM 1/256, α 0.2 (paper figs 0.15). Higher = failure (native).
- **Impl.** [`safe.py`](safe.py): `SafeDetector(variant=mlp|lstm)`, `requires={hidden}`,
  `supervised=True`. `fit(labeled_episodes)` consumes `(episode_records, success_bool)` pairs and trains
  the exact SAFE objective (MLP push-down/push-up on per-step sigmoids; LSTM BCE on `1−success`),
  inverse-class-freq + L2, variable-length sequences padded per batch. `score(rec)` returns the per-step
  signal (MLP `p_t`; LSTM 1-step); the faithful full-history LSTM score is `score_stream(recs)` (which the
  harness uses for whole episodes — SAFE-LSTM is the only detector that overrides `score_stream`). The
  cumulative-sum (MLP `s_t`) + functional-CP band are the **shared harness's** aggregation+threshold, so
  the per-chunk score stays threshold-free like every other baseline. Synthetic-verified: both variants
  AUROC 1.0 separating labeled success/failure; fit-without-labels and score-before-fit raise.
- 🔶 **Adapted (flagged), all valid:** (1) SAFE uses **only trajectory-level labels** broadcast to every
  step + a time-weight (MIL/weak supervision — the paper does not need the failure timestep); we broadcast
  the outcome identically (true per-step labels would be an ablation, not the headline). (2) The
  feature-reduction (which layer/norm, horizon/flow pooling) is **config-driven + grid-searched** in SAFE
  (best of First/Last/Mean/First&Last per benchmark); we fix the PizeroDatasetConfig default (mean/mean)
  in the GPU `hidden` producer, alternatives exposed as `--horizon-reduce`/`--diff-reduce`. (3) Time-weight
  `get_time_weight` and the LSTM hard-negative term are omitted (both off by default in the repo).
- ⚠️ **Two hand-offs (needs both):** (a) the **`hidden` capture stage** — `cli/hidden_states.py` /
  `pipelines/hidden_states.py` re-runs the FM head at the recorded context with a forward-pre-hook on
  `action_out_proj` (torch.compile **must be OFF** or the hook silently misses), storing the reduced
  `(d,)` feature (`supports_hidden=True` + `embed_context_hidden`); (b) **FAILURE labels** — SAFE
  is supervised, so `detector_score` routes it through a labeled fit split spanning **both** success and
  failure rollouts (the unsupervised success-only family stays separate). Until a fresh
  `--record-context` eval + the hidden stage are run, SAFE is synthetic-verified only.

---

## 6. Reproduction notes & known gaps (read before reporting numbers)

1. **Normalization scale (accel / straightness / oracle).** The **post-hoc `detector_score` path scores
   each chunk in isolation**, so `accel`/`straightness`/`oracle` **self-normalize per chunk** and do
   **not** land on the offline per-episode-/run-pooled label scale. `run_detector_score` builds
   detectors with bare defaults, so there is currently no way to inject a fixed/run-pooled std there.
   This is why an end-to-end smoke run scored through that path is on the
   **self-norm** scale and is **not** the π₀.₅ paper numbers (0.78–0.88). To reproduce the validated
   numbers, run the **online server path** (which records `chunk_accels`, optionally with
   `--accel-fixed-std`) or thread a fixed std through the post-hoc detectors.
2. **GPU hand-offs (synthetic-verified only until run):** (a) the **obs-embedding capture hook**
   (`cli/obs_emb.py`) unlocks `rnd_oe`/`logpzo`/`pca_kmeans`/`knn`/`mahalanobis`/`fiper`; (b) the
   **action-expert-hidden capture stage** (`cli/hidden_states.py`, `supports_hidden` +
   `embed_context_hidden`, torch.compile OFF) unlocks `safe` — which **also** needs FAILURE labels (its
   fit spans both success and failure rollouts, unlike the success-only family); (c) the **per-chunk
   `fm_velocity`** (via `ChunkResampleSession`) unlocks `fm_loss`. `stac` needs **no** new hook — it
   reads the same `chunk_divergence` resamples as `ace`/`oracle` (the previous decision's set is threaded
   forward as `prev_resample` by `detector_score`), so it is scoreable today. `detector_score` drops any
   detector whose required signal is unavailable and **logs a warning naming the dropped detectors + their
   missing signals**.
3. **The published "alarm" lives in the harness, not the detector** — sliding-window aggregation +
   success-calibrated conformal/quantile thresholds (FIPER window-SUM, FAIL-Detect functional CP band,
   Diff-DAgger 0.99-quantile + K-window). Our main ranking is raw-stream AUROC
   to stay α-independent; a paper's headline TPR@FPR is the *banded* detector, not the raw score.
4. **Success-only `fit` must use closed-loop SUCCESS rollouts, not demos** (demos are straighter and
   mis-calibrate the scale — the `accel_normalization_scale` lesson). Keep calibrate / train / test
   splits disjoint; never calibrate a CP/quantile threshold on the test split.

### Fidelity status at a glance

All marks below reflect the post-fix state.

| detector | source | vs paper | vs official repo | resolution |
|---|---|---|---|---|
| `accel` | project | n/a | n/a | default self-norm ≠ validated scale → documented (pass `fixed_std`); exec-mode `n_exec` guard added |
| `straightness` | project | n/a | n/a | self-norm caveat documented |
| `sparc` | Balasubramanian 2015 | 🔶 adapted | ✅ faithful helper | ✓ attribution fixed (not FAIL-Detect); `fc` band documented |
| `ace` | FIPER | ✅ | ✅ | ✓ `fit` calibrates per-dim cell + adaptive grid + NaN mask; `pos_dims` documented |
| `stac` | Sentinel / FAIL-Detect | ✅ | ✅ Sentinel / 🔶 FAIL-Detect | ✓ biased V-stat + median-of-squared γ + overlap slice + gripper-drop; full-chunk fallback = FAIL-Detect-style (flagged); restored + `prev_resample` re-added |
| `oracle_resample_spread` | project GT | n/a | n/a | ✓ self-std removed; run-pooled `dim_scale` wired (magnitude preserved) |
| `fm_loss` | Diff-DAgger | 🔶 adapted | n/a (own re-impl) | ✓ velocity contract `−v_θ(x_t,1−t)` documented; task-id in seed |
| `rnd_oe` | RND / FIPER / FAIL-Detect | ✅ | ✅ | ✓ asymmetric predictor + orthogonal init + no obs norm |
| `logpzo` | FAIL-Detect | ✅ | ✅ | ✓ training flipped to data→noise (matches repo); AUROC 1.0 |
| `pca_kmeans` | FAIL-Detect | ✅ | ✅ | ✓ raw full-dim PCA + KMeans(64) nearest-centroid |
| `knn` / `mahalanobis` | Sun 2022 / Lee 2018 | ✅ | n/a (not in repo) | ✓ knn L2-normalized; attribution corrected |
| `fiper` | FIPER | ✅ | 🔶 adapted | ✓ calibrates ACE in `fit` + positive threshold floor (window/CP-band still in harness) |
| `safe` | SAFE | ✅ | ✅ | ✓ MLP push-down/up + LSTM BCE on last-hidden; mean/mean reduce; cumsum + functional-CP in harness; needs `hidden` hook + FAILURE labels |
| `base/registry/harness` | project | n/a | n/a | ✓ `REQUIREABLE` comment corrected; `detector_score` warns on dropped detectors; `prev_resample` + `score_stream` + `supervised` added |

> **Note.** `fiper` stays 🔶 vs the repo because its published *alarm* (sliding-window SUM + functional
> conformal band) lives in the shared harness by design, not in the detector — the per-chunk score is the
> threshold-free `min` of the two calibrated legs. The two previously-high-severity bugs (`logpzo`
> orientation, `oracle` scale) are **fixed and re-verified**.
