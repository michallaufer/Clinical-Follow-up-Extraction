import os
import json
import torch
import gc
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from src.data_utils import load_and_split_data

# --- Configuration ---
MODEL_NAME = "unsloth/llama-3-8b-instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "models/llama-3-finetuned-results"

SYSTEM_PROMPT = """You are an expert clinical information extraction system.

Task:
Extract ONLY scheduled follow-up items from the clinical note (tests/labs/imaging/referrals/therapy explicitly planned for the future).
Ignore history/past tests/symptom duration.

Return ONLY valid JSON.
"""

def _safe_load_actions_gt(x):
    if not isinstance(x, str) or not x.strip(): return []
    try: return json.loads(x)
    except: return []

def format_row(row, allowed_actions):
    """Formats a single row into the LLaMA-3 Instruct prompt template."""
    note = row.get("note_text", "")
    gt_list = _safe_load_actions_gt(row.get("actions_gt", "[]"))
    
    # Target is strictly [{"action": act, "period_date": pd}]
    target = []
    for a in gt_list:
        act = a.get("action")
        pd = a.get("period_date")
        if act and pd:
            target.append({"action": act, "period_date": pd})

    allowed_str = "\n".join([f"- {a}" for a in allowed_actions])

    user_prompt = f"""Rules:
- Output must be a JSON array (possibly empty).
- Each item must be a scheduled follow-up item from the note (not history).
- action must be EXACTLY one of the allowed actions below (no new actions).
- period_date must be computed relative to the visit date line in the note (YYYY-MM-DD).
- If no scheduled items: return [].
- Output JSON only (no markdown, no extra text).

Allowed actions:
{allowed_str}

Clinical note:
{note}
"""
    target_json = json.dumps(target, ensure_ascii=False)

    text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{target_json}<|eot_id|>"""

    return {"text": text}

def main():
    print("1. Loading datasets and applying 3-way split...")
    df_train, df_val, df_test, info = load_and_split_data("data/synthetic_clinical_notes_2000.csv")
    
    # Gather the closed-set actions to pass into the prompt
    all_closed_actions = sorted(list(set(info["train_actions"] + info["val_actions"] + info["test_actions"])))

    print("2. Formatting HuggingFace Datasets...")
    train_dataset = Dataset.from_pandas(df_train).map(lambda r: format_row(r, all_closed_actions), remove_columns=df_train.columns)
    val_dataset = Dataset.from_pandas(df_val).map(lambda r: format_row(r, all_closed_actions), remove_columns=df_val.columns)

    print(f"3. Loading FastLanguageModel ({MODEL_NAME})...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = None,
        load_in_4bit = True,
    )

    print("4. Attaching LoRA Adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ_LENGTH,
        packing = True, 
        args = TrainingArguments(
            output_dir = OUTPUT_DIR,
            per_device_train_batch_size = 8,
            gradient_accumulation_steps = 2,
            learning_rate = 2e-4,
            warmup_steps = 50,
            num_train_epochs = 4,
            logging_steps = 10,
            eval_strategy = "steps",
            eval_steps = 50,
            save_steps = 50,
            save_total_limit = 2,
            fp16 = False,
            bf16 = True,
            tf32 = True,
            optim="adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            report_to = "none",
            group_by_length = True,
        ),
    )

    print("5. Starting LLaMA-3 LoRA Finetuning...")
    trainer.train()

    print(f"6. Saving final model and adapters to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    # Clean up GPU memory before starting
    gc.collect()
    torch.cuda.empty_cache()
    main()