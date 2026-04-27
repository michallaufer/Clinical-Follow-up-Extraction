import numpy as np

def calculate_set_f1(pred_set, gold_set):
    """Calculates Exact Match F1 for sets of tuples."""
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    return p, r, f1

def score_single_note(pred_items, gold_items):
    """
    Expects lists of dicts: [{'action': 'MRI', 'days_offset': 14}]
    Returns counts for calculating F1 and MAE later.
    """
    p_acts = {p['action'] for p in pred_items}
    g_acts = {g['action'] for g in gold_items}
    
    p_pairs = {(p['action'], p['days_offset']) for p in pred_items}
    g_pairs = {(g['action'], g['days_offset']) for g in gold_items}

    # Diagnostics for MAE (Mean Absolute Error)
    abs_errors = []
    p_map = {p['action']: p['days_offset'] for p in pred_items}
    g_map = {g['action']: g['days_offset'] for g in gold_items}
    
    common_acts = p_map.keys() & g_map.keys()
    for act in common_acts:
        abs_errors.append(abs(p_map[act] - g_map[act]))

    return {
        "act_tp": len(p_acts & g_acts),
        "act_fp": len(p_acts - g_acts),
        "act_fn": len(g_acts - p_acts),
        "pair_tp": len(p_pairs & g_pairs),
        "pair_fp": len(p_pairs - g_pairs),
        "pair_fn": len(g_pairs - p_pairs),
        "abs_errors": abs_errors
    }

def bootstrap_confidence_intervals(all_note_scores, n_iterations=1000, seed=123):
    """
    Performs note-level bootstrap resampling.
    all_note_scores: List of dicts returned by score_single_note()
    """
    rng = np.random.default_rng(seed)
    n_notes = len(all_note_scores)
    
    bootstrapped_metrics = {"action_f1": [], "pair_f1": [], "mae": []}
    
    for _ in range(n_iterations):
        # Resample notes with replacement
        indices = rng.integers(0, n_notes, size=n_notes)
        sample_scores = [all_note_scores[i] for i in indices]
        
        # Aggregate counts
        act_tp = sum(s["act_tp"] for s in sample_scores)
        act_fp = sum(s["act_fp"] for s in sample_scores)
        act_fn = sum(s["act_fn"] for s in sample_scores)
        
        pair_tp = sum(s["pair_tp"] for s in sample_scores)
        pair_fp = sum(s["pair_fp"] for s in sample_scores)
        pair_fn = sum(s["pair_fn"] for s in sample_scores)
        
        all_errors = []
        for s in sample_scores:
            all_errors.extend(s["abs_errors"])
            
        _, _, act_f1 = calculate_set_f1(set(), set()) # Using raw counts instead
        act_p = act_tp / (act_tp + act_fp) if (act_tp + act_fp) > 0 else 0.0
        act_r = act_tp / (act_tp + act_fn) if (act_tp + act_fn) > 0 else 0.0
        act_f1_score = 2 * act_p * act_r / (act_p + act_r) if (act_p + act_r) > 0 else 0.0
        
        pair_p = pair_tp / (pair_tp + pair_fp) if (pair_tp + pair_fp) > 0 else 0.0
        pair_r = pair_tp / (pair_tp + pair_fn) if (pair_tp + pair_fn) > 0 else 0.0
        pair_f1_score = 2 * pair_p * pair_r / (pair_p + pair_r) if (pair_p + pair_r) > 0 else 0.0
        
        bootstrapped_metrics["action_f1"].append(act_f1_score)
        bootstrapped_metrics["pair_f1"].append(pair_f1_score)
        bootstrapped_metrics["mae"].append(np.mean(all_errors) if all_errors else 0.0)

    # Calculate Mean and 95% CI bounds
    results = {}
    for metric, values in bootstrapped_metrics.items():
        arr = np.array(values)
        results[metric] = {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, 2.5)),
            "ci_upper": float(np.percentile(arr, 97.5))
        }
        
    return results