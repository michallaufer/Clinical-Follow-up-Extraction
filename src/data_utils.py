import pandas as pd
import numpy as np
import ast
import json
import os

# -------------------------
# Configuration Settings
# -------------------------
VAL_OOV_K = 4
TEST_OOV_K = 6
BASE_SEED = 123
MAX_TRIES = 50

# Minimum-size checks
MIN_TRAIN_NOTES = 800
MIN_VAL_NOTES   = 150
MIN_TEST_NOTES  = 150
MIN_TRAIN_ENTS  = 800
MIN_VAL_ENTS    = 150
MIN_TEST_ENTS   = 150

# -------------------------
# Helper Functions
# -------------------------
def parse_actions_gt(s):
    """Safely parse the ground truth JSON/string into a list."""
    if s is None or (isinstance(s, float) and np.isnan(s)) or s == "":
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        pass
    return []

def row_actions_set(row):
    """Return set of canonical action labels (strings) appearing in a note."""
    items = row.get("actions_gt_obj", [])
    out = set()
    for it in items:
        a = it.get("action")
        if a is not None and str(a).strip() != "":
            out.add(str(a).strip())
    return out

def count_action_entities(df_):
    """Total number of action entities (not unique types) across notes."""
    return int(df_["actions_gt_obj"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0).sum())

def build_three_way_split(df_, seed):
    """Builds a disjoint split where Validation and Test actions are unseen in Train."""
    all_actions = sorted({
        it.get("action")
        for xs in df_["actions_gt_obj"]
        for it in (xs if isinstance(xs, list) else [])
        if it.get("action") is not None and str(it.get("action")).strip() != ""
    })
    all_actions = [str(a).strip() for a in all_actions]
    all_actions = sorted(set(all_actions))

    if len(all_actions) < (VAL_OOV_K + TEST_OOV_K + 1):
        raise ValueError(f"Not enough actions for the split. Found {len(all_actions)} actions.")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_actions))
    test_actions = [all_actions[i] for i in perm[:TEST_OOV_K]]
    val_actions  = [all_actions[i] for i in perm[TEST_OOV_K:TEST_OOV_K+VAL_OOV_K]]
    train_actions = [a for a in all_actions if a not in set(test_actions) and a not in set(val_actions)]

    TEST_SET = set(test_actions)
    VAL_SET  = set(val_actions)
    TRAIN_SET = set(train_actions)

    actions_sets = df_.apply(row_actions_set, axis=1)

    def is_subset(s, allowed):
        return s.issubset(allowed)

    is_train = actions_sets.apply(lambda s: is_subset(s, TRAIN_SET))  
    is_val   = actions_sets.apply(lambda s: (len(s) > 0) and is_subset(s, VAL_SET))
    is_test  = actions_sets.apply(lambda s: (len(s) > 0) and is_subset(s, TEST_SET))

    df_train = df_[is_train].copy()
    df_val   = df_[is_val].copy()
    df_test  = df_[is_test].copy()

    info = {
        "seed": seed,
        "all_actions_n": len(all_actions),
        "train_actions_n": len(TRAIN_SET),
        "val_actions_n": len(VAL_SET),
        "test_actions_n": len(TEST_SET),
        "train_notes": len(df_train),
        "val_notes": len(df_val),
        "test_notes": len(df_test),
        "train_ents": count_action_entities(df_train),
        "val_ents": count_action_entities(df_val),
        "test_ents": count_action_entities(df_test),
        "train_actions": sorted(TRAIN_SET),
        "val_actions": sorted(VAL_SET),
        "test_actions": sorted(TEST_SET),
    }
    return df_train, df_val, df_test, info

def split_passes_min_checks(info):
    return (
        info["train_notes"] >= MIN_TRAIN_NOTES and
        info["val_notes"]   >= MIN_VAL_NOTES and
        info["test_notes"]  >= MIN_TEST_NOTES and
        info["train_ents"]  >= MIN_TRAIN_ENTS and
        info["val_ents"]    >= MIN_VAL_ENTS and
        info["test_ents"]   >= MIN_TEST_ENTS
    )

# -------------------------
# Main Execution Function
# -------------------------
def load_and_split_data(data_path):
    """
    Loads the clinical notes dataset, parses the ground truth, and returns 
    the 3-way disjoint split required for the research paper.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find dataset at {data_path}")
        
    df = pd.read_csv(data_path, keep_default_na=False)
    df["actions_gt_obj"] = df["actions_gt"].apply(parse_actions_gt)

    best = None
    for k in range(MAX_TRIES):
        seed = BASE_SEED + k
        df_train_seen, df_val_oov, df_test_oov, info = build_three_way_split(df, seed)
        if split_passes_min_checks(info):
            best = (df_train_seen, df_val_oov, df_test_oov, info)
            break

    if best is None:
        print("WARNING: Minimum-size checks not met. Using last attempted seed.")
        df_train_seen, df_val_oov, df_test_oov, info = build_three_way_split(df, BASE_SEED + MAX_TRIES - 1)
    else:
        df_train_seen, df_val_oov, df_test_oov, info = best

    return df_train_seen, df_val_oov, df_test_oov, info

# If you run this file directly, it prints the summary.
if __name__ == "__main__":
    # Point this to wherever you uploaded the CSV in Cursor
    test_path = "../data/synthetic_clinical_notes_2000.csv" 
    
    try:
        train, val, test, split_info = load_and_split_data(test_path)
        print("\n=== 3-way disjoint action split summary ===")
        print(f"Seed used: {split_info['seed']}")
        print(f"Total canonical actions: {split_info['all_actions_n']}")
        print(f"Train notes: {split_info['train_notes']} | Val-OOV notes: {split_info['val_notes']} | Test-OOV notes: {split_info['test_notes']}")
        print("\nVal-OOV actions (4):", split_info["val_actions"])
        print("Test-OOV actions (6):", split_info["test_actions"])
    except Exception as e:
        print(e)