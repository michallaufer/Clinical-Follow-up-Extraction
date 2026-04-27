import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set global style for academic figures
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def load_metrics(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}

def plot_f1_comparisons(bert_metrics, llama_strict, llama_constrained, gpt_metrics):
    """Generates the grouped bar chart for F1 Scores."""
    
    # Standardize data structures
    metrics_by_model = {
        "BioBERT pipeline": bert_metrics,
        "LLaMA (Strict)": llama_strict,
        "LLaMA (Constrained)": llama_constrained,
        "ChatGPT zero-shot": gpt_metrics,
    }

    metric_keys = [
        ("STRICT_action_f1", "Action F1"),  # Maps to your respective keys
        ("STRICT_date_only_f1", "Date Only F1"),
        ("STRICT_action_date_f1", "Action+Date Pair F1"),
    ]

    models = list(metrics_by_model.keys())
    labels = [m[1] for m in metric_keys]

    # Map keys robustly based on your notebook's fallback mappings
    def get_f1(model_dict, key):
        # Handle BioBERT's different key naming
        if "ner_span_TEST_f1" in model_dict and key == "STRICT_action_f1":
            return model_dict.get("ner_span_TEST_f1", 0)
        if "ner_span_TIME_f1" in model_dict and key == "STRICT_date_only_f1":
            return model_dict.get("ner_span_TIME_f1", 0)
        if "action_date_f1" in model_dict and key == "STRICT_action_date_f1":
            return model_dict.get("action_date_f1", 0)
        return model_dict.get(key, 0.0)

    vals = np.array([[get_f1(metrics_by_model[model], k) for (k, _) in metric_keys] for model in models], dtype=float)

    x = np.arange(len(labels))
    bar_w = 0.8 / len(models)

    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#ffbb78', '#2ca02c'] 

    for i, model_name in enumerate(models):
        plt.bar(x + i*bar_w - 0.4 + bar_w/2, vals[i], width=bar_w, label=model_name, color=colors[i])

    plt.xticks(x, labels, rotation=0, fontsize=11)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1.05)
    plt.ylabel("F1 Score", fontsize=12)
    plt.title("Model Comparison: BioBERT vs. LLaMA vs. ChatGPT", fontsize=14, weight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("models/f1_comparison.png", dpi=300)
    print("Saved F1 Comparison Plot to models/f1_comparison.png")

def plot_date_mae(bert_metrics, llama_strict, gpt_metrics):
    """Generates the Date Calculation Accuracy (MAE) Plot."""
    models = ["BioBERT", "LLaMA (Strict)", "ChatGPT"]
    
    def get_mae(d):
        return d.get("period_date_abs_err_days_mae_on_matched_actions", 0.0)
        
    maes = [get_mae(bert_metrics), get_mae(llama_strict), get_mae(gpt_metrics)]
    
    plt.figure(figsize=(8, 5))
    colors = ['#2ecc71' if x < 2.0 else '#e74c3c' for x in maes]
    ax = sns.barplot(x=models, y=maes, palette=colors)

    plt.title("Date Calculation Accuracy (Mean Absolute Error)", fontsize=16, weight='bold')
    plt.ylabel("Avg Error in Days (Lower is Better)")
    
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.1, f'{height:.1f} Days', ha="center", weight='bold')

    plt.tight_layout()
    plt.savefig("models/date_mae.png", dpi=300)
    print("Saved MAE Plot to models/date_mae.png")

if __name__ == "__main__":
    print("Loading metrics...")
    # Adjust paths if you named the output files differently in main.py
    bert_metrics = load_metrics("models/biobert_final_metrics.json")
    llama_strict = load_metrics("models/llama_finetuned_metrics.json")
    gpt_metrics = load_metrics("models/gpt_metrics.json") # Assuming you save GPT metrics here

    if bert_metrics and llama_strict:
        plot_f1_comparisons(bert_metrics, llama_strict, llama_strict, gpt_metrics)
        plot_date_mae(bert_metrics, llama_strict, gpt_metrics)
    else:
        print("Could not find metric JSON files in the /models directory. Run the evaluation scripts first.")