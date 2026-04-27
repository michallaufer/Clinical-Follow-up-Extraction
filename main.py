import asyncio
import os
import pandas as pd
from unsloth import FastLanguageModel
from openai import AsyncOpenAI

from src.data_utils import load_and_split_data
from src.inference import BioBERTExtractor, LlamaExtractor, ChatGPTExtractor
from src.evaluate import score_single_note, bootstrap_confidence_intervals
from src.ontology import normalize_action
from src.time_utils import time_text_to_days_offset

def standardize_ground_truth(raw_gt_list, visit_date):
    """Converts the dataset's raw ground truth into the uniform testing format."""
    standardized = []
    for gt in raw_gt_list:
        canon = normalize_action(gt.get("action", ""))
        days = time_text_to_days_offset(gt.get("period_text", ""))
        if canon and days is not None:
            standardized.append({"action": canon, "days_offset": int(days)})
    return standardized

# 1. Define the async batch function outside of main()
async def evaluate_chatgpt_batch(df_test, gpt_extractor):
    sem = asyncio.Semaphore(20) # Concurrency limit
    
    async def process_row(row):
        async with sem:
            note_text = row.get("note_text", "")
            visit_date = row.get("visit_date", "")
            if not note_text or not visit_date: 
                return []
            
            return await gpt_extractor.predict_async(note_text, visit_date)
            
    # Gather creates a list of predictions in the exact same order as the dataframe rows
    tasks = [process_row(row) for idx, row in df_test.iterrows()]
    return await asyncio.gather(*tasks)


def main():
    print("=== Clinical Follow-Up Extraction Evaluation ===")
    
    print("\n1. Loading datasets...")
    _, _, df_test, split_info = load_and_split_data("data/synthetic_clinical_notes_2000.csv")
    print(f"Test-OOV Notes: {len(df_test)}")
    
    # ==================================================
    # LLaMA PIPELINE
    # ==================================================
    print("\n2. Initializing LLaMA Extractor...")
    llama_model, llama_tokenizer = FastLanguageModel.from_pretrained(
        model_name="models/llama-3-finetuned-results", 
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(llama_model) 
    llama_extractor = LlamaExtractor(model=llama_model, tokenizer=llama_tokenizer)

    print("\n3. Running LLaMA Inference and Scoring...")
    llama_note_scores = []
    
    for idx, row in df_test.iterrows():
        note_text = row.get("note_text", "")
        visit_date = row.get("visit_date", "")
        
        if not note_text or not visit_date:
            continue
            
        pred_items = llama_extractor.predict(note_text, visit_date)
        raw_gold_items = row.get("actions_gt_obj", []) 
        gold_items = standardize_ground_truth(raw_gold_items, visit_date)
        
        score_dict = score_single_note(pred_items, gold_items)
        llama_note_scores.append(score_dict)

    # ==================================================
    # CHATGPT PIPELINE
    # ==================================================
    print("\n4. Initializing ChatGPT Extractor...")
    # Ensure OPENAI_API_KEY is exported in your terminal environment
    client = AsyncOpenAI() 
    gpt_extractor = ChatGPTExtractor(client)

    print("\n5. Running ChatGPT Inference (Async)...")
    # Execute the async function synchronously and get the ordered list of predictions
    gpt_predictions = asyncio.run(evaluate_chatgpt_batch(df_test, gpt_extractor))
    
    print("   Scoring ChatGPT Outputs...")
    gpt_note_scores = []
    
    # Zip the predictions together with the dataframe rows to score them
    for pred_items, (idx, row) in zip(gpt_predictions, df_test.iterrows()):
        visit_date = row.get("visit_date", "")
        raw_gold_items = row.get("actions_gt_obj", []) 
        gold_items = standardize_ground_truth(raw_gold_items, visit_date)
        
        score_dict = score_single_note(pred_items, gold_items)
        gpt_note_scores.append(score_dict)

    # ==================================================
    # EVALUATION AND OUTPUT
    # ==================================================
    print("\n6. Calculating Bootstrap Confidence Intervals (N=1000)...")
    llama_results = bootstrap_confidence_intervals(llama_note_scores, n_iterations=1000)
    gpt_results = bootstrap_confidence_intervals(gpt_note_scores, n_iterations=1000)
    
    print("\n=== FINAL RESULTS (LLaMA-3 LoRA Pipeline) ===")
    print(f"Action-Only F1 : {llama_results['action_f1']['mean']:.3f} (95% CI: {llama_results['action_f1']['ci_lower']:.3f} - {llama_results['action_f1']['ci_upper']:.3f})")
    print(f"Strict Pair F1 : {llama_results['pair_f1']['mean']:.3f} (95% CI: {llama_results['pair_f1']['ci_lower']:.3f} - {llama_results['pair_f1']['ci_upper']:.3f})")
    print(f"Offset MAE     : {llama_results['mae']['mean']:.2f} days (95% CI: {llama_results['mae']['ci_lower']:.2f} - {llama_results['mae']['ci_upper']:.2f})")

    print("\n=== FINAL RESULTS (ChatGPT Zero-Shot) ===")
    print(f"Action-Only F1 : {gpt_results['action_f1']['mean']:.3f} (95% CI: {gpt_results['action_f1']['ci_lower']:.3f} - {gpt_results['action_f1']['ci_upper']:.3f})")
    print(f"Strict Pair F1 : {gpt_results['pair_f1']['mean']:.3f} (95% CI: {gpt_results['pair_f1']['ci_lower']:.3f} - {gpt_results['pair_f1']['ci_upper']:.3f})")
    print(f"Offset MAE     : {gpt_results['mae']['mean']:.2f} days (95% CI: {gpt_results['mae']['ci_lower']:.2f} - {gpt_results['mae']['ci_upper']:.2f})")

if __name__ == "__main__":
    main()