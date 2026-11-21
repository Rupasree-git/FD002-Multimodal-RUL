# ============================================================
# CONTRADICTION DETECTION MODULE
# ============================================================
# Detects disagreement between modality predictions
# Used for uncertainty-aware decision support
# ============================================================

import numpy as np


def detect_contradictions(modality_preds, threshold=50):
    """
    Detect contradictions between modalities.

    Parameters
    ----------
    modality_preds : dict
        Dictionary of modality predictions.
        Example:
        {
            "sensor": np.array([...]),
            "visual": np.array([...]),
            "tabular": np.array([...])
        }

    threshold : float
        Standard deviation threshold (cycles) above which
        modalities are considered contradictory.

    Returns
    -------
    contradictions : np.ndarray (bool)
        True if disagreement > threshold for that sample

    disagreement : np.ndarray (float)
        Standard deviation across modality predictions
    """
    if not modality_preds:
        raise ValueError("modality_preds must be a non-empty dictionary.")

    arrays = [np.asarray(v, dtype=np.float64) for v in modality_preds.values()]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"All modality prediction arrays must have the same length; got {lengths}.")

    # Shape -> (num_modalities, num_samples)
    preds_array = np.vstack(arrays)

    # Disagreement across modalities
    disagreement = np.std(preds_array, axis=0)

    # Contradiction flag
    contradictions = disagreement > threshold

    return contradictions, disagreement