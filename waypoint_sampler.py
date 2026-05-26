"""Waypoint samplers for H-LeWM Stage-2 training.

A *waypoint sampler* picks which frames in a T-frame training window become
"waypoints" — the anchors between which action chunks get compressed into a
single macro-action.

Three schemes are implemented here. They correspond to the three HWM-paper
backbones described in
``plans/hlewm_train_test_methodology.md`` → Scheme 4 / 1 / 2:

    sample_waypoints
        random interior frames, fixed full-T window — CURRENT default,
        kept verbatim. (No paper precedent — see issues/issues.md.)

    sample_waypoints_fixed_stride
        deterministic, evenly-spaced waypoints across the full T-frame window.
        Matches PLDM (HWM Appendix B.4, the JEPA-from-pixels backbone — the
        closest analog to LeWM).

    sample_waypoints_variable_span
        randomise a sub-window of length L ~ Uniform(L_min, L_max) inside the
        full T-frame window, then place N+2 evenly-spaced waypoints inside.
        Matches DINO-WM (HWM Appendix B.3).

All three return a sorted 1-D LongTensor of shape (N+2,) containing valid
frame indices in [0, T-1] — endpoints always included. So they're drop-in
replacements for each other from the consumer's perspective.

`N` here means the count of **interior** waypoints. Total waypoints = N + 2.
Number of segments between consecutive waypoints = N + 1.
"""

import torch


# ══════════════════════════════════════════════════════════════════════════════
#  Scheme 4 (current default) — random interior frames, no minimum gap
# ══════════════════════════════════════════════════════════════════════════════


def sample_waypoints(T: int, N: int = 3, device=None) -> torch.Tensor:
    """Return sorted waypoint frame indices, with N interior frames chosen
    uniformly at random (no minimum gap).

    This is the current Stage-2 default. It has no direct paper precedent
    (see issues/issues.md). Provided here unchanged for backward compat.

    Parameters
    ----------
    T      : trajectory window length (number of frames)
    N      : number of *interior* waypoints (total returned = N + 2)
    device : torch device for the returned tensor

    Returns
    -------
    1-D LongTensor of shape (N+2,) — sorted, starts at 0, ends at T-1.
    """
    if N >= T - 1:
        # very short trajectory — just use every frame
        return torch.arange(T, device=device)
    interior = torch.randperm(T - 2, device=device)[:N] + 1   # never 0 or T-1
    endpoints = torch.tensor([0, T - 1], device=device)
    return torch.cat([endpoints, interior]).sort().values


# ══════════════════════════════════════════════════════════════════════════════
#  Scheme 1 — PLDM-style fixed stride (deterministic)
# ══════════════════════════════════════════════════════════════════════════════


def sample_waypoints_fixed_stride(T: int, N: int = 3, device=None) -> torch.Tensor:
    """Return evenly-spaced waypoint frame indices across the full T-frame window.

    Matches the PLDM recipe from HWM Appendix B.4
    (*"extract N waypoint states using a fixed stride"*). Deterministic — every
    call with the same T, N returns the same indices, so within-sample variance
    in segment length is zero (or off-by-one when T-1 is not evenly divisible
    by N+1).

    Parameters
    ----------
    T      : trajectory window length (number of frames)
    N      : number of *interior* waypoints (total returned = N + 2)
    device : torch device for the returned tensor

    Returns
    -------
    1-D LongTensor of shape (N+2,) — sorted, starts at 0, ends at T-1.

    Examples
    --------
    >>> sample_waypoints_fixed_stride(20, N=3)
    tensor([ 0,  5, 10, 14, 19])
    >>> sample_waypoints_fixed_stride(60, N=4)   # PLDM-style: 6 waypoints, stride 12
    tensor([ 0, 12, 24, 36, 48, 59])
    """
    if N >= T - 1:
        return torch.arange(T, device=device)
    # linspace + round preserves endpoints exactly and spaces interior as evenly
    # as integer indices allow.
    idx = torch.linspace(0, T - 1, N + 2, device=device).round().long()
    return idx


# ══════════════════════════════════════════════════════════════════════════════
#  Scheme 2 — DINO-WM-style variable-span sub-window
# ══════════════════════════════════════════════════════════════════════════════


def sample_waypoints_variable_span(
    T: int,
    N: int = 3,
    L_min: int | None = None,
    L_max: int | None = None,
    device=None,
) -> torch.Tensor:
    """Pick a random sub-window of length L ~ Uniform(L_min, L_max) inside
    the T-frame window, then place N+2 evenly-spaced waypoints inside.

    Matches the DINO-WM recipe from HWM Appendix B.3
    (*"subsample trajectory segments with lengths uniformly drawn between
    25 and 70 timesteps. From each segment, we sample N waypoint states"*).

    Within-sample variance in segment length is small (at most 1 frame, like
    fixed-stride). Across samples, the *total* span varies, which gives the
    model a controlled range of macro-action durations.

    Parameters
    ----------
    T      : full trajectory window length (number of frames)
    N      : number of *interior* waypoints (total returned = N + 2)
    L_min  : minimum sub-window length, in frames. Default: max(N + 2, T // 2).
    L_max  : maximum sub-window length, in frames. Default: T.
    device : torch device for the returned tensor

    Returns
    -------
    1-D LongTensor of shape (N+2,) — sorted, frame indices in [0, T-1].
    Note: indices need NOT include 0 or T-1 — they span the random sub-window,
    which may start anywhere in the T-frame window.

    Raises
    ------
    ValueError if T is too small to fit N+2 waypoints, or if L_min > L_max,
    or if L_max > T, or if L_min < N + 2.
    """
    if N + 2 > T:
        raise ValueError(f"T={T} too small for N+2={N+2} waypoints")

    if L_min is None:
        L_min = max(N + 2, T // 2)
    if L_max is None:
        L_max = T
    if L_min < N + 2:
        raise ValueError(f"L_min={L_min} must be >= N+2={N+2}")
    if L_max > T:
        raise ValueError(f"L_max={L_max} must be <= T={T}")
    if L_min > L_max:
        raise ValueError(f"L_min={L_min} must be <= L_max={L_max}")

    # Draw the sub-window length and start offset.
    L = int(torch.randint(L_min, L_max + 1, (1,), device=device).item())
    s = int(torch.randint(0, T - L + 1, (1,), device=device).item())

    # Place N+2 evenly-spaced waypoints inside [s, s + L - 1] using the same
    # linspace+round trick as fixed_stride.
    idx = torch.linspace(s, s + L - 1, N + 2, device=device).round().long()
    return idx


# ══════════════════════════════════════════════════════════════════════════════
#  Convenience dispatcher (optional)
# ══════════════════════════════════════════════════════════════════════════════


_SCHEMES = {
    "random":         sample_waypoints,                  # current default
    "fixed_stride":   sample_waypoints_fixed_stride,     # PLDM
    "variable_span":  sample_waypoints_variable_span,    # DINO-WM
}


def sample_waypoints_by_name(name: str, T: int, N: int = 3, **kwargs) -> torch.Tensor:
    """Dispatch by scheme name. Convenient for config-driven selection.

    name ∈ {"random", "fixed_stride", "variable_span"}.

    >>> sample_waypoints_by_name("fixed_stride", 20, N=3)
    tensor([ 0,  5, 10, 14, 19])
    """
    if name not in _SCHEMES:
        raise ValueError(f"unknown waypoint scheme {name!r}; choose from {sorted(_SCHEMES)}")
    return _SCHEMES[name](T, N=N, **kwargs)
