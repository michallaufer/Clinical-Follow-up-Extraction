# Clinical Follow-Up Extraction

This repository contains code for extracting scheduled clinical follow-up instructions from free-text outpatient notes.

The project formulates follow-up extraction as a structured clinical information extraction task. Instead of directly generating JSON from a note, the proposed pipeline decomposes the task into:

1. **TestSpecification extraction** — identify scheduled clinical actions such as MRI, CT Scan, Blood Test, or specialist consult.
2. **TimeSpecification extraction** — identify timing expressions such as "in 2 weeks", "tomorrow", or "in 3 months".
3. **ScheduledFor relation extraction** — link each TestSpecification entity to the TimeSpecification entity that defines when it should occur.
4. **Normalization** — map action mentions to canonical labels and convert timing expressions into deterministic `days_offset` values relative to the visit date.

The main model is a BioBERT-based structured information extraction pipeline with a token-level NER head and a learned entity-linking module. It is compared against two generative baselines:

- zero-shot GPT-4o-mini
- fine-tuned LLaMA-3 LoRA

## Project Motivation

Clinical follow-up instructions are often written only in free-text notes, for example:

Schedule MRI in 2 weeks and repeat blood work in 3 months.

The target structured output is:

[
  {"action": "MRI", "days_offset": 14},
  {"action": "Blood Test", "days_offset": 90}
]

Extracting this information enables downstream systems to track whether recommended follow-up tests, procedures, referrals, or monitoring actions were completed on time.

## Task Definition

**Input**

```text
clinical note text + visit date
```

**Output**

```json
[
  {"action": "canonical TestSpecification", "days_offset": 14}
]
```

Intermediate structured representation:

TestSpecification entity
TimeSpecification entity
ScheduledFor(TestSpecification, TimeSpecification)

## Dataset

The benchmark contains 2,000 synthetic outpatient-style notes generated from controlled structured skeletons.

The generation process varies:

- clinical domain
- clinical scenario
- number of scheduled follow-up items
- TestSpecification label
- TimeSpecification phrasing
- plan-header style
- note layout
- linguistic style
- distractor temporal context

The dataset is synthetic and does not contain real patient data.

The dataset is synthetic and does not contain real patient data.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── main.py
├── run_gpu.py
├── src/
│   ├── data_gen.py
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── ontology.py
│   ├── plots.py
│   ├── time_utils.py
│   ├── train_biobert.py
│   └── train_llama.py
├── data/
│   ├── README.md
│   └── synthetic_clinical_notes.csv
├── models/
│   └── README.md
└── results/
    ├── README.md
    ├── seen_vs_oov_comparison.png
    ├── biobert_ner_token_report_seen.csv
    └── biobert_ner_token_report_oov.csv
```

## Installation

Create a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```
## Environment Variables

For GPT-4o-mini evaluation:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

For BioBERT evaluation:

```bash
export BIOBERT_MODEL_DIR="models/biobert_joint_seen_oov"
```

Do not commit .env files or API keys.

Running the Project

The main entry point is:

python main.py

For GPU / RunPod execution:

python -u run_gpu.py

Depending on the local setup, additional scripts in src/ may be used for training, inference, evaluation, and plotting.

Main Evaluation Metrics

The project reports:

TestSpecification F1 — correct canonical action labels
TimeSpecification Offset F1 — correct normalized day offsets
Test-Time Pair F1 — correct complete (action, days_offset) pair
Time offset MAE — mean absolute error in days on matched actions
Bootstrap 95% confidence intervals by note-level resampling
Notes on Model Checkpoints

Model weights are not included in this repository by default..

##Citation

If using this code, please cite the accompanying project report:

Reliable Extraction of Clinical Follow-Up Instructions as Structured Information Extraction.
Disclaimer

This repository is for research purposes only. It is not a clinical decision-support system and should not be used for patient care without validation on real institution-specific clinical notes and appropriate clinical review.
