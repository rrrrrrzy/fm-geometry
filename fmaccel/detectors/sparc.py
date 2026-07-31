"""``sparc`` — spectral arc length of the planned chunk's action speed profile.

SPARC (Balasubramanian et al., 2015) is a dimensionless movement-**smoothness** metric: the
arc length of the *normalized Fourier magnitude spectrum* of a speed profile. A smooth
movement concentrates energy at low frequency → short arc (value near 0); a jerky one
spreads energy → long arc (more negative). We take the executed action window of the recorded
chunk endpoint ``x_t[-1]``, build its speed profile $\\lVert\\Delta a_t\\rVert$ over chunk
positions (per-dim standardized so dims weigh comparably), and return $-\\mathrm{SPARC}$ so the
score follows the accel convention (**higher = jerkier = more likely failure**). Free, online,
training-free, pure numpy.

A standard movement-smoothness metric; here a free geometric comparator distinct from accel — accel
reads the *denoise-path* bend, SPARC reads the *executed-action* jerk over chunk time. (NOTE: SPARC is
**not** one of FAIL-Detect's score functions — those are logpZO / RND / CFM / PCA_kmeans / NatPN / DER;
do not attribute SPARC to FAIL-Detect.)

⚠️ CAVEAT (short-signal): SPARC is designed for a *rest-to-rest* movement and a single action
chunk (~16 positions → ~15 speed samples) is short for a spectral arc length, so a perfectly
constant-velocity chunk can leak and read as non-smooth. The discrimination among realistic
chunks (smooth < jerky) holds, but the *faithful* SPARC form computes the spectrum over the
executed-action **stream / trailing window** of the rollout — add that windowed variant via the
post-hoc detector_score path (it has the full executed-action history); per-chunk SPARC here is
the cheap online approximation, and its empirical AUROC is the test of whether it suffices.

⚠️ BAND: with ``fs=1.0`` (one chunk position = one step) and ``fc=0.5`` = Nyquist, the whole spectrum
up to Nyquist is kept (``fc/Nyquist = 1.0``), unlike the reference SPARC's band-limiting ``fc`` (e.g.
``fc=10`` Hz at a real ~100 Hz sampling rate, ``fc/Nyquist ≈ 0.2``). Reasonable for a ~15-sample signal
but more sensitive to the leakage-prone high-frequency end; ``amp_th`` still trims low-amplitude bins.
"""

from __future__ import annotations

import numpy as np

from fmaccel.detectors.base import ChunkRecord, Detector


def spectral_arc_length(
    speed: np.ndarray, *, fs: float = 1.0, padlevel: int = 4, fc: float = 0.5, amp_th: float = 0.05
) -> float:
    """SPARC of a 1-D speed profile (Balasubramanian 2015).

    Returns the (negative) arc length of the normalized magnitude spectrum over ``[0, fc]``,
    restricted to the band where the normalized magnitude is ``>= amp_th``. 0 (least negative)
    = smoothest. Returns ``0.0`` for a degenerate (empty / constant / single-bin) profile.
    ``fs=1.0`` treats one chunk position as one time step (Nyquist 0.5).
    """
    speed = np.asarray(speed, np.float64)
    if speed.size < 2 or not np.any(np.abs(speed) > 0):
        return 0.0
    nfft = int(2 ** (np.ceil(np.log2(speed.size)) + padlevel))
    mag = np.abs(np.fft.rfft(speed, n=nfft))
    peak = mag.max()
    if peak <= 0:
        return 0.0
    mag = mag / peak
    freq = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = freq <= fc
    f_sel, m_sel = freq[band], mag[band]
    above = np.where(m_sel >= amp_th)[0]
    if above.size < 2:
        return 0.0
    f_sel = f_sel[above[0] : above[-1] + 1]
    m_sel = m_sel[above[0] : above[-1] + 1]
    span = f_sel[-1] - f_sel[0]
    if span <= 0:
        return 0.0
    df = np.diff(f_sel) / span
    dm = np.diff(m_sel)
    return float(-np.sum(np.sqrt(df ** 2 + dm ** 2)))


class SparcDetector(Detector):
    """$-\\mathrm{SPARC}$ of the executed chunk's action speed profile (higher = jerkier = failure)."""

    name = "sparc"
    requires = frozenset({"x_t"})
    online = True

    def __init__(self, *, n_exec: int | None = None, fc: float = 0.5, amp_th: float = 0.05) -> None:
        self.n_exec = n_exec
        self.fc = float(fc)
        self.amp_th = float(amp_th)

    def score(self, rec: ChunkRecord) -> float:
        self.check(rec)
        x = np.asarray(rec.x_t, np.float32)  # (T+1, chunk, act)
        a = x[-1, :, : rec.action_dim]  # planned/executed chunk endpoint (chunk, act)
        n_exec = rec.n_exec if rec.n_exec is not None else self.n_exec
        if n_exec is not None:
            a = a[: int(n_exec)]
        if a.shape[0] < 3:  # need a few positions for a meaningful spectrum
            return 0.0
        sd = a.std(axis=0) + 1e-8
        speed = np.linalg.norm(np.diff(a / sd, axis=0), axis=1)  # (n-1,) per-dim-standardized speed
        sal = spectral_arc_length(speed, fc=self.fc, amp_th=self.amp_th)
        return -float(sal)  # accel convention: higher = jerkier = more likely failure
