# Models

Model checkpoints are not committed to this repository because they may be large.

Expected local model directories:

models/biobert_joint_seen_oov/
models/llama-3-seen-oov-results/
models/eval_seen_oov/

The BioBERT directory should contain files such as:

pytorch_model.bin
model_config.json
tokenizer.json
tokenizer_config.json

The LLaMA directory should contain the trained LoRA adapter files.

Evaluation outputs are written under:

models/eval_seen_oov/
