"""Size bins, COCO S/M/L mapping and box-to-patch-grid helpers.

See docs/CONTRACTS.md ("Size bins" and "Feature store"). Everything here is pure
numpy and deterministic.
"""
from __future__ import annotations

import numpy as np

#: Right-open bin edges on ``max_box_frac`` (relative bbox area of the largest instance).
#: Bin ``b`` is ``[BIN_EDGES[b], BIN_EDGES[b + 1])``; the last bin also includes 1.0.
BIN_EDGES: list[float] = [0, 0.001, 0.003, 0.01, 0.03, 0.10, 1.0]
#: Human-readable names, one per bin.
BIN_NAMES: list[str] = ["<0.1%", "0.1-0.3%", "0.3-1%", "1-3%", "3-10%", ">10%"]
#: Number of size bins.
N_BINS: int = len(BIN_NAMES)

#: COCO small/medium/large names, indexed by the codes returned by :func:`coco_sml`.
SML_NAMES: list[str] = ["small", "medium", "large"]
#: COCO area thresholds in original pixels: small < 32^2, medium < 96^2, else large.
SML_EDGES: list[float] = [32.0 ** 2, 96.0 ** 2]

#: ViT patch size of the DINOv2 backbones.
PATCH_SIZE: int = 14


def assign_bins(max_box_frac: np.ndarray) -> np.ndarray:
    """Map relative areas to size-bin indices.

    Parameters
    ----------
    max_box_frac
        Array of any shape with values in ``[0, 1]``. ``0`` means the class is absent.

    Returns
    -------
    np.ndarray
        ``int8`` array of the same shape: bin index ``0..5`` where ``frac > 0``
        (right-open intervals; exactly ``1.0`` falls in the last bin) and ``-1``
        where ``frac == 0``.

    Notes
    -----
    Edges are compared in the input's float dtype so that a ``float32`` value
    that *is* an edge (e.g. ``np.float32(0.01)``) lands in the bin that starts
    at that edge, as it would in ``float64``.
    """
    frac = np.asarray(max_box_frac)
    dtype = frac.dtype if frac.dtype.kind == "f" else np.float64
    frac = frac.astype(dtype, copy=False)
    edges = np.asarray(BIN_EDGES, dtype=dtype)
    # searchsorted(side="right") - 1 gives b with edges[b] <= frac < edges[b+1].
    idx = np.searchsorted(edges, frac, side="right") - 1
    idx = np.clip(idx, 0, N_BINS - 1)  # frac == 1.0 (or > 1) -> last bin
    idx = idx.astype(np.int8)
    idx[frac <= 0] = -1
    return idx


def coco_sml(max_seg_frac: np.ndarray, width: np.ndarray, height: np.ndarray) -> np.ndarray:
    """COCO small/medium/large codes from segmentation-area fractions.

    Parameters
    ----------
    max_seg_frac
        ``(N, 80)`` (or ``(N,)``) relative segmentation area of the largest instance;
        ``0`` means absent.
    width, height
        ``(N,)`` original image size in pixels. They are broadcast against the
        leading axis of ``max_seg_frac``.

    Returns
    -------
    np.ndarray
        ``int8`` array shaped like ``max_seg_frac``: ``0`` small (area < 32^2),
        ``1`` medium (area < 96^2), ``2`` large, ``-1`` absent. Area is measured in
        original pixels: ``frac * width * height``.
    """
    frac = np.asarray(max_seg_frac, dtype=np.float64)
    w = np.asarray(width, dtype=np.float64)
    h = np.asarray(height, dtype=np.float64)
    if frac.ndim > w.ndim:
        extra = (1,) * (frac.ndim - w.ndim)
        w = w.reshape(w.shape + extra)
        h = h.reshape(h.shape + extra)
    area = frac * w * h
    out = np.where(area < SML_EDGES[0], 0, np.where(area < SML_EDGES[1], 1, 2)).astype(np.int8)
    out[frac <= 0] = -1
    return out


def patch_area_frac(res: int, patch: int = PATCH_SIZE) -> float:
    """Fraction of the image covered by one patch token at input resolution ``res``.

    ``res = 224, patch = 14`` gives a 16x16 grid, so one patch is ``1/256``.
    """
    grid = res / patch
    return 1.0 / (grid * grid)


def _centers(grid: int) -> np.ndarray:
    return (np.arange(grid, dtype=np.float64) + 0.5) / grid


def box_to_patch_mask(box_xyxy: np.ndarray, grid: int) -> np.ndarray:
    """Boolean mask of the patches whose centre lies inside a normalised box.

    Parameters
    ----------
    box_xyxy
        ``(4,)`` box ``(x0, y0, x1, y1)`` normalised to ``[0, 1]``.
    grid
        Patches per side ``G``; the token grid is ``G x G`` row-major from the top-left.

    Returns
    -------
    np.ndarray
        ``(G*G,)`` bool, row-major (index ``r * G + c``). A patch ``(r, c)`` is inside
        if its centre ``((c+0.5)/G, (r+0.5)/G)`` lies in ``[x0, x1) x [y0, y1)``. If no
        centre is inside (sub-patch object) the single patch containing the box centre
        is marked instead, so the mask is never empty.
    """
    x0, y0, x1, y1 = (float(v) for v in np.asarray(box_xyxy, dtype=np.float64).reshape(4))
    c = _centers(grid)
    in_x = (c >= x0) & (c < x1)
    in_y = (c >= y0) & (c < y1)
    mask = in_y[:, None] & in_x[None, :]  # (rows, cols)
    if not mask.any():
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        col = int(np.clip(np.floor(cx * grid), 0, grid - 1))
        row = int(np.clip(np.floor(cy * grid), 0, grid - 1))
        mask[row, col] = True
    return mask.reshape(-1)


def boxes_to_patch_mask(boxes_xyxy: np.ndarray, grid: int) -> np.ndarray:
    """Union of :func:`box_to_patch_mask` over ``(k, 4)`` boxes; all-False when ``k == 0``."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    mask = np.zeros(grid * grid, dtype=bool)
    for box in boxes:
        mask |= box_to_patch_mask(box, grid)
    return mask
