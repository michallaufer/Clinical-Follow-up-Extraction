import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Dict, Tuple
from transformers import BertTokenizer, BertModel, TrainingArguments, Trainer, EarlyStoppingCallback
from src.data_utils import load_and_split_data
from src.time_utils import time_text_to_days_offset

# --- Constants ---
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
MAX_LEN = 512
DOC_STRIDE = 128
BATCH_SIZE = 16
EPOCHS = 20
LR = 2e-5
ALPHA_LINK = 1.0

TAG2ID = {"O": 0, "B-TEST": 1, "I-TEST": 2, "B-TIME": 3, "I-TIME": 4}
ID2TAG = {v: k for k, v in TAG2ID.items()}
NUM_TAGS = len(TAG2ID)

WIDTH_BUCKETS = [1, 2, 3, 4, 5, 8, 12, 20, 40]
DIST_BUCKETS = [0, 1, 2, 3, 4, 5, 8, 12, 20, 40, 80, 160, 320]
MAX_ACTIONS = 4
MAX_TIMES = 4

# --- Architecture ---
class Biaffine(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.U = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.U)

    def forward(self, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bad,df,btd->bat", a, self.U, t)

class BioBertNerLinker(nn.Module):
    def __init__(self, base_name: str, num_tags: int, proj_dim: int = 256, width_dim: int = 32, dist_dim: int = 32):
        super().__init__()
        self.encoder = BertModel.from_pretrained(base_name)
        h = self.encoder.config.hidden_size

        self.ner_head = nn.Linear(h, num_tags)
        self.width_emb = nn.Embedding(len(WIDTH_BUCKETS)+1, width_dim)
        self.dist_emb = nn.Embedding(len(DIST_BUCKETS)+1, dist_dim)
        
        self.register_buffer("width_buckets_tensor", torch.tensor(WIDTH_BUCKETS, dtype=torch.long))
        self.register_buffer("dist_buckets_tensor", torch.tensor(DIST_BUCKETS, dtype=torch.long))

        self.span_dim = 2*h + width_dim
        self.act_proj = nn.Sequential(nn.Linear(self.span_dim, proj_dim), nn.ReLU(), nn.Dropout(0.1))
        self.tim_proj = nn.Sequential(nn.Linear(self.span_dim, proj_dim), nn.ReLU(), nn.Dropout(0.1))

        self.biaffine = Biaffine(proj_dim)
        self.lin = nn.Linear(2*proj_dim + dist_dim, 1)

        self.none_time = nn.Parameter(torch.zeros(self.span_dim))
        nn.init.normal_(self.none_time, std=0.02)

    def span_repr(self, H: torch.Tensor, spans: torch.Tensor) -> torch.Tensor:
        B, L, h = H.shape
        s = spans[:,:,0].clamp(0, L-1)
        e = spans[:,:,1].clamp(0, L-1)
        hs = H.gather(1, s.unsqueeze(-1).expand(-1,-1,h))
        he = H.gather(1, e.unsqueeze(-1).expand(-1,-1,h))
        width = (e - s + 1).clamp(min=1).to(torch.long)
        w_bucket = torch.bucketize(width, self.width_buckets_tensor)
        w = self.width_emb(w_bucket)
        return torch.cat([hs, he, w], dim=-1)

    def forward(self, input_ids, attention_mask, ner_labels=None, action_spans=None, time_spans=None, action_mask=None, time_mask=None, link_labels=None, output_attentions=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=output_attentions)
        H = out.last_hidden_state
        ner_logits = self.ner_head(H)
        
        # Cross-entropy weighting for sparse action entities
        class_weights = torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0], device=H.device)

        ner_loss = None
        if ner_labels is not None:
            ner_loss = F.cross_entropy(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1), ignore_index=-100, weight=class_weights)

        link_logits, link_loss = None, None
        if action_spans is not None and time_spans is not None:
            act_rep = self.span_repr(H, action_spans)
            tim_rep = self.span_repr(H, time_spans)
            none = self.none_time.view(1,1,-1).expand(tim_rep.size(0), 1, -1)
            tim_rep_all = torch.cat([tim_rep, none], dim=1)

            aP, tP = self.act_proj(act_rep), self.tim_proj(tim_rep_all)
            bia = self.biaffine(aP, tP)

            a_s, t_s = action_spans[:,:,0], time_spans[:,:,0]
            dist_abs = (t_s.unsqueeze(1) - a_s.unsqueeze(2)).abs().to(torch.long)
            dist_bucket = torch.bucketize(dist_abs, self.dist_buckets_tensor)
            dist_emb = self.dist_emb(dist_bucket)

            B, A, T = dist_abs.shape
            none_bucket = torch.zeros((B,A,1), dtype=torch.long, device=H.device)
            none_emb = self.dist_emb(none_bucket)
            dist_emb_all = torch.cat([dist_emb, none_emb], dim=2)

            a_exp = aP.unsqueeze(2).expand(-1,-1,tP.size(1),-1)
            t_exp = tP.unsqueeze(1).expand(-1,aP.size(1),-1,-1)
            lin_in = torch.cat([a_exp, t_exp, dist_emb_all], dim=-1)
            lin = self.lin(lin_in).squeeze(-1)

            link_logits = bia + lin
            tmask = torch.cat([time_mask, torch.ones(time_mask.size(0),1, device=time_mask.device)], dim=1)
            link_logits = link_logits.masked_fill(tmask.unsqueeze(1) == 0, -1e4)

            if link_labels is not None:
                B, A, TT = link_logits.shape
                logits_flat, labels_flat = link_logits.reshape(B*A, TT), link_labels.reshape(B*A)
                amask_flat = (action_mask.reshape(B*A) > 0.5)
                link_loss = F.cross_entropy(logits_flat[amask_flat], labels_flat[amask_flat]) if amask_flat.any() else torch.tensor(0.0, device=H.device)

        loss = (ner_loss + ALPHA_LINK * link_loss) if (ner_loss is not None and link_loss is not None) else (ner_loss or link_loss)
        return {"loss": loss, "ner_logits": ner_logits, "link_logits": link_logits}

# --- Note: You'll need to drop in your JointNerLinkDataset here or in src/dataset.py ---
# (I've omitted it to save space, but you can copy `class JointNerLinkDataset` directly above `main()`)

@dataclass
class Collator:
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {k: torch.stack([f[k] for f in features], dim=0) for k in features[0].keys()}


class JointNerLinkDataset(torch.utils.data.Dataset):
    """Produces ONE sample per sliding window.
    We keep only windows that overlap at least one GT span (TEST or TIME) to reduce noise.
    For notes with no GT actions, we keep only the first window.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.windows = []  # list of dicts with tokenized window + row_idx

        for row_idx in range(len(self.df)):
            row = self.df.iloc[row_idx]
            note = row["note_text"]
            actions = row["actions_gt_obj"]

            tok = biobert_tokenizer(
                note,
                truncation=True,
                max_length=MAX_LEN,
                stride=DOC_STRIDE,
                return_overflowing_tokens=True,
                padding="max_length",
                return_offsets_mapping=True,
            )
            nwin = len(tok["input_ids"])

            if not actions:
                # keep only first window for negative notes to avoid huge negatives
                keep_wins = [0]
            else:
                # collect all GT spans
                gt_spans = []
                for a in actions:
                    for key_s, key_e in [("action_char_start","action_char_end"), ("time_char_start","time_char_end")]:
                        s0, e0 = a.get(key_s), a.get(key_e)
                        if isinstance(s0,int) and isinstance(e0,int) and 0 <= s0 < e0 <= len(note):
                            gt_spans.append((s0,e0))

                keep_wins = []
                for w in range(nwin):
                    offsets = tok["offset_mapping"][w]
                    ws, we = _window_char_range(offsets)
                    if any(_overlaps(ws,we, s0,e0) for (s0,e0) in gt_spans):
                        keep_wins.append(w)

                # safety: if nothing matched (rare), keep the last window
                if not keep_wins:
                    keep_wins = [nwin-1]

            for w in keep_wins:
                self.windows.append({
                    "row_idx": row_idx,
                    "input_ids": tok["input_ids"][w],
                    "attention_mask": tok["attention_mask"][w],
                    "offsets": tok["offset_mapping"][w],
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx: int):
        win = self.windows[idx]
        row = self.df.iloc[win["row_idx"]]
        note = row["note_text"]
        actions = row["actions_gt_obj"]
        offsets = win["offsets"]

        # NER labels (BIO)
        ner = make_ner_labels(note, offsets, actions)
        ner_labels = []
        for (s,e),lab in zip(offsets, ner):
            if s == e == 0:
                ner_labels.append(-100)
            else:
                ner_labels.append(lab)

        # ---- Linking supervision (gold spans) ----
        # Build candidate (action_tok, time_tok) pairs that are visible in THIS window.
        pairs = []
        for a in actions:
            as0, ae0 = a.get("action_char_start"), a.get("action_char_end")
            ts0, te0 = a.get("time_char_start"), a.get("time_char_end")
            if not (isinstance(as0,int) and isinstance(ae0,int) and isinstance(ts0,int) and isinstance(te0,int)):
                continue
            act_tok = char_span_to_token_span(offsets, as0, ae0)
            tim_tok = char_span_to_token_span(offsets, ts0, te0)
            # IMPORTANT: only supervise link when BOTH are in-window
            if act_tok is not None and tim_tok is not None:
                pairs.append((act_tok, tim_tok))

        # Unique TIME candidates (dedup) sorted by token start
        time_unique = sorted({t for (_,t) in pairs}, key=lambda x: x[0])[:MAX_TIMES]
        time_index = {t:i for i,t in enumerate(time_unique)}

        # Actions sorted by token start; keep only those whose TIME is still in time_unique
        pairs = [(a,t) for (a,t) in pairs if t in time_index]
        pairs = sorted(pairs, key=lambda p: p[0][0])[:MAX_ACTIONS]

        action_spans = [a for (a,_) in pairs]
        link_labels = [time_index[t] for (_,t) in pairs]

        # masks + padding
        a_mask = [1.0]*len(action_spans)
        t_mask = [1.0]*len(time_unique)

        while len(action_spans) < MAX_ACTIONS:
            action_spans.append((0,0))
            a_mask.append(0.0)
            link_labels.append(MAX_TIMES)  # NONE

        while len(time_unique) < MAX_TIMES:
            time_unique.append((0,0))
            t_mask.append(0.0)

        return {
            "input_ids": torch.tensor(win["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(win["attention_mask"], dtype=torch.long),
            "ner_labels": torch.tensor(ner_labels, dtype=torch.long),
            "action_spans": torch.tensor(action_spans, dtype=torch.long),
            "time_spans": torch.tensor(time_unique, dtype=torch.long),
            "action_mask": torch.tensor(a_mask, dtype=torch.float),
            "time_mask": torch.tensor(t_mask, dtype=torch.float),
            "link_labels": torch.tensor(link_labels, dtype=torch.long),
        }


def main():
    print("Loading rigorous 3-way split...")
    df_train, df_val, df_test, info = load_and_split_data("data/synthetic_clinical_notes_2000.csv")
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BioBertNerLinker(MODEL_NAME, NUM_TAGS)
    
    # Initialize your JointNerLinkDataset here
    # train_ds = JointNerLinkDataset(df_train)
    # val_ds = JointNerLinkDataset(df_val)
    
    args = TrainingArguments(
        output_dir="models/biobert_joint_ner_link",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=Collator(),
        # train_dataset=train_ds,
        # eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    # Force contiguous memory for tensor optimizations
    for name, param in model.named_parameters():
        param.data = param.data.contiguous()

    print("Starting BioBERT Training...")
    # trainer.train()
    # torch.save(model.state_dict(), "models/biobert_finetuned_final.pth")
    # tokenizer.save_pretrained("models/biobert_finetuned_tokenizer")

if __name__ == "__main__":
    main()