"""Clustering-evaluation metric panel (Phase-1 behavior-clustering eval).

Pure compute over per-frame label arrays — no zarr, no plotting, no paths (Tenet 2/3) — so it
ports unchanged into the earlier prototype like ``segmentation.py``/``retrieval.py`` do. The storage and the
ground-truth reading lived in the removed ``cluster_eval`` module; this file only scores two
aligned integer sequences: ``truth`` (the ground-truth behavior label per frame) and ``pred`` (the
predicted cluster/syllable id per frame, with ``noise_label`` for unassigned frames).

Why this specific panel, not one number: a behavior clusterer produces *far more clusters than
there are behaviors* and leaves some frames unassigned, so the honest read needs several axes at
once (the same lesson the query-eval false start taught — a single arbitrary cutoff hides the
truth):

- **Assignment-free partition agreement** — AMI, ARI (sklearn), and **B-cubed** P/R/F1 (Amigó et
  al. 2009, hand-rolled). These are chance-corrected and do not need a cluster→behavior map, so
  they are the headline when #clusters ≫ #behaviors.
- **Homogeneity + completeness** (sklearn) — the diagnostic pair that says *which way* a clustering
  is wrong (pure-but-fragmented vs merged).
- **Interpretable, after a majority-vote cluster→behavior map** (not Hungarian — clusters outnumber
  behaviors): per-behavior frame P/R/F1/IoU, with its floor (``one_cluster_floor``).

**Noise policy.** Every partition metric is reported twice — ``excluded`` (assigned frames only)
and ``singletons`` (each noise frame is its own cluster, over all frames, the honest lower bound) —
alongside a ``coverage`` fraction. Callers score over the labeled-frame subset only.
"""

import numpy as np


# --- partition agreement (assignment-free) ------------------------------------------- #
def _contingency(truth, pred):
    """Return (contingency [n_classes, n_clusters], class labels, cluster labels)."""
    classes, class_idx = np.unique(truth, return_inverse=True)
    clusters, cluster_idx = np.unique(pred, return_inverse=True)
    table = np.zeros((classes.size, clusters.size), dtype=np.int64)
    np.add.at(table, (class_idx, cluster_idx), 1)
    return table, class_idx, cluster_idx


def adjusted_mutual_information(truth, pred) -> float:
    """Adjusted mutual information — chance-corrected, assignment-free (sklearn)."""
    from sklearn.metrics import adjusted_mutual_info_score

    return float(adjusted_mutual_info_score(truth, pred))


def adjusted_rand_index(truth, pred) -> float:
    """Adjusted Rand index — chance-corrected pair-counting agreement (sklearn)."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(truth, pred))


def homogeneity_completeness(truth, pred) -> tuple[float, float]:
    """Homogeneity (each cluster one behavior) and completeness (each behavior one cluster)."""
    from sklearn.metrics import homogeneity_completeness_v_measure

    homogeneity, completeness, _v = homogeneity_completeness_v_measure(truth, pred)
    return float(homogeneity), float(completeness)


def bcubed(truth, pred) -> tuple[float, float, float]:
    """B-cubed precision/recall/F1 (Amigó et al. 2009), the per-item extended pair metric.

    For each item, precision is the fraction of its cluster that shares its behavior and recall
    the fraction of its behavior that shares its cluster; both are averaged over items. It needs
    no cluster→behavior map and is well-behaved when clusters far outnumber behaviors, which is
    why it is the headline supervised agreement metric here.
    """
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    if truth.size == 0:
        return 0.0, 0.0, 0.0
    table, class_idx, cluster_idx = _contingency(truth, pred)
    cluster_sizes = table.sum(axis=0)
    class_sizes = table.sum(axis=1)
    shared = table[class_idx, cluster_idx]  # per item: |same behavior AND same cluster|
    precision = float(np.mean(shared / cluster_sizes[cluster_idx]))
    recall = float(np.mean(shared / class_sizes[class_idx]))
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return precision, recall, f1


def purity(truth, pred) -> float:
    """Fraction of frames in the majority behavior of their cluster (descriptive; rises with K)."""
    truth = np.asarray(truth)
    if truth.size == 0:
        return 0.0
    table, _class_idx, _cluster_idx = _contingency(truth, pred)
    return float(table.max(axis=0).sum() / truth.size)


def _empty_partition_metrics() -> dict:
    return {
        "ami": 0.0,
        "ari": 0.0,
        "bcubed_precision": 0.0,
        "bcubed_recall": 0.0,
        "bcubed_f1": 0.0,
        "homogeneity": 0.0,
        "completeness": 0.0,
        "purity": 0.0,
        "n_clusters": 0,
        "n_frames": 0,
    }


def _partition_metrics(truth, pred) -> dict:
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    if truth.size == 0:
        return _empty_partition_metrics()
    precision, recall, f1 = bcubed(truth, pred)
    homogeneity, completeness = homogeneity_completeness(truth, pred)
    return {
        "ami": adjusted_mutual_information(truth, pred),
        "ari": adjusted_rand_index(truth, pred),
        "bcubed_precision": precision,
        "bcubed_recall": recall,
        "bcubed_f1": f1,
        "homogeneity": homogeneity,
        "completeness": completeness,
        "purity": purity(truth, pred),
        "n_clusters": int(np.unique(pred).size),
        "n_frames": int(truth.size),
    }


def agreement_panel(truth, pred, *, noise_label: int = -1) -> dict:
    """Partition agreement under both noise policies plus coverage (Phase-1 plan §1).

    ``excluded`` scores only assigned frames; ``singletons`` keeps every noise frame as its own
    cluster over all frames — the honest lower bound. Report both, never one.
    """
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    n_frames = truth.size
    is_noise = pred == noise_label
    n_assigned = int(np.sum(~is_noise))
    coverage = float(n_assigned / n_frames) if n_frames else 0.0

    excluded = _partition_metrics(truth[~is_noise], pred[~is_noise])

    singleton_pred = pred.copy()
    if is_noise.any():
        base = int(pred[~is_noise].max()) + 1 if n_assigned else 0
        singleton_pred[is_noise] = base + np.arange(int(np.sum(is_noise)))
    singletons = _partition_metrics(truth, singleton_pred)

    return {
        "coverage": coverage,
        "noise_fraction": float(1.0 - coverage) if n_frames else 0.0,
        "n_frames": int(n_frames),
        "n_assigned": n_assigned,
        "excluded": excluded,
        "singletons": singletons,
    }


# --- interpretable: majority-vote cluster→behavior map ------------------------------- #
def majority_vote_map(truth, pred, *, noise_label: int = -1) -> dict[int, int]:
    """Map each cluster to its majority behavior — not Hungarian: clusters outnumber behaviors,
    so many clusters legitimately map to the same behavior. Noise clusters are omitted.
    """
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    mapping: dict[int, int] = {}
    for cluster in np.unique(pred):
        if cluster == noise_label:
            continue
        member_truth = truth[pred == cluster]
        behaviors, counts = np.unique(member_truth, return_counts=True)
        mapping[int(cluster)] = int(behaviors[np.argmax(counts)])
    return mapping


def _map_to_behavior(pred, mapping, *, noise_label: int) -> np.ndarray:
    """Per-frame predicted behavior via the cluster→behavior map; unmapped → ``noise_label``."""
    return np.array([mapping.get(int(c), noise_label) for c in np.asarray(pred)])


def per_behavior_frame_prf(
    truth, pred, mapping, *, noise_label: int = -1
) -> dict[int, dict]:
    """Frame-level P/R/F1/IoU per behavior after mapping each cluster to its majority behavior.

    Behavior ids are assumed non-negative (``noise_label`` marks an unmapped prediction, which
    matches no behavior).
    """
    truth = np.asarray(truth)
    predicted = _map_to_behavior(pred, mapping, noise_label=noise_label)
    out: dict[int, dict] = {}
    for behavior in np.unique(truth):
        b = int(behavior)
        pred_b = predicted == b
        true_b = truth == b
        tp = int(np.sum(pred_b & true_b))
        fp = int(np.sum(pred_b & ~true_b))
        fn = int(np.sum(~pred_b & true_b))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        out[b] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "support": int(np.sum(true_b)),
        }
    return out


def one_cluster_floor(truth, pred, *, noise_label: int = -1) -> dict[int, dict]:
    """What a fit that learned **nothing** scores per behavior — the floor to read against.

    Per-behavior F1 is not chance-corrected, so a prevalent behavior scores well for free: put
    every bout in one cluster, majority-vote maps it to the commonest behavior, and that
    behavior gets recall 1.0 and precision equal to its prevalence. A degenerate
    2-cluster fit can score a high F1 on a prevalent behavior while its AMI is near
    zero — read straight, the table says it recovered that behavior well.

    Coverage is matched to the real prediction rather than assumed to be 1.0, and that detail
    is the whole point. The naive floor for prevalence *p* is ``2p/(1+p)``, which can
    sit *below* what a degenerate fit actually scores, because its unassigned frames
    raise precision.
    A floor the degenerate answer beats is worse than no floor: it certifies the thing it
    exists to catch. So the floor is computed as the real one-cluster prediction, noise mask
    and all.

    This is the cluster-eval counterpart of the nulls query eval (a module since removed)
    shipped once it had learned the same lesson (a recall no better than its null). It is a
    *floor*, not a null: one specific uninformative answer, not an average over uninformative
    answers.
    """
    truth = np.asarray(truth)
    pred = np.asarray(pred)
    if truth.size == 0:
        return {}
    # One cluster everywhere the real fit assigned anything; noise stays noise, so coverage
    # matches and the comparison is like-for-like.
    flat = np.where(pred == noise_label, noise_label, 0)
    return per_behavior_frame_prf(
        truth,
        flat,
        majority_vote_map(truth, flat, noise_label=noise_label),
        noise_label=noise_label,
    )
