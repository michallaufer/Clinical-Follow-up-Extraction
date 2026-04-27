import json
from .ontology import normalize_action, normalize_action_with_expansion, ACTION_ONTOLOGY
from .time_utils import time_text_to_days_offset
from datetime import datetime

class ClinicalExtractor:
    """Base class to guarantee uniform outputs across all models."""
    def predict(self, note_text: str, visit_date: str) -> list:
        raise NotImplementedError

class BioBERTExtractor(ClinicalExtractor):
    def __init__(self, model, tokenizer, device, max_len=512, doc_stride=128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.doc_stride = doc_stride
        self.model.eval()

    # NOTE: Keep your `char_span_to_token_span`, `decode_spans_from_tags`, and 
    # `predict_ner_spans_windowed` helper methods inside this class.
    def char_span_to_token_span(offsets: List[Tuple[int,int]], char_s: int, char_e: int):
        """Map a character span [char_s, char_e) to an inclusive token span (tok_s, tok_e).
        Returns None if the span is fully outside the current window (e.g., truncated).
        """
        tok_ids = []
        for i,(s,e) in enumerate(offsets):
            if s == e == 0:
                continue
            if s < char_e and e > char_s:  # overlap
                tok_ids.append(i)
        if not tok_ids:
            return None
        return min(tok_ids), max(tok_ids)  # inclusive

    
    def decode_spans_from_tags(offsets, tag_ids: List[int], prefix: str, note_text: str):
        spans = []
        cur = None
        for i,(off,tid) in enumerate(zip(offsets, tag_ids)):
            s,e = off
            if s == e == 0:
                continue
            tag = ID2TAG.get(int(tid), "O")
            if tag == f"B-{prefix}":
                if cur is not None:
                    spans.append(cur)
                cur = [i,i]
            elif tag == f"I-{prefix}":
                if cur is not None:
                    cur[1] = i
            else:
                if cur is not None:
                    spans.append(cur); cur=None
        if cur is not None:
            spans.append(cur)

        out=[]
        for ts,te in spans:
            cs = offsets[ts][0]
            ce = offsets[te][1]
            txt = note_text[cs:ce].strip()
            if txt:
                out.append((cs,ce,txt,ts,te))
        return out

    @torch.no_grad()
def predict_ner_spans_windowed(model, tokenizer, note_text: str, max_len=512, doc_stride=128):
    """
    Sliding-window NER inference:
    - runs BioBERT on overlapping windows
    - decodes spans per window
    - converts them to global char spans
    - deduplicates
    Returns: act_spans, tim_spans
    Each span format stays compatible with your decode_spans_from_tags output.
    """
    model.eval()

    # Tokenize into overflowing windows
    enc = tokenizer(
        note_text,
        truncation=True,
        max_length=max_len,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]          # [num_windows, max_len]
    attention_mask = enc["attention_mask"]
    offsets_all = enc["offset_mapping"]   # [num_windows, max_len, 2]
    # maps each window to the original example (single text => all 0)
    # overflow_to_sample_mapping = enc["overflow_to_sample_mapping"]

    all_act = []
    all_tim = []

    for w in range(input_ids.size(0)):
        ids = input_ids[w:w+1].to(DEVICE)
        am  = attention_mask[w:w+1].to(DEVICE)
        offsets = offsets_all[w].tolist()

        out = model(input_ids=ids, attention_mask=am)
        tag_ids = out["ner_logits"].argmax(-1).squeeze(0).detach().cpu().tolist()

        act_w = decode_spans_from_tags(offsets, tag_ids, "TEST", note_text)
        tim_w = decode_spans_from_tags(offsets, tag_ids, "TIME", note_text)

        # decode_spans_from_tags already uses global offsets from note_text,
        # because offsets are absolute char positions in the original string.
        all_act.extend(act_w)
        all_tim.extend(tim_w)

    # Deduplicate spans by char start/end
    def dedup(spans):
        seen = set()
        out = []
        for s in spans:
            cs, ce = s[0], s[1]
            key = (cs, ce)
            if key not in seen:
                seen.add(key)
                out.append(s)
        # sort by start position for stability
        out.sort(key=lambda x: (x[0], x[1]))
        return out

    return dedup(all_act), dedup(all_tim)

    def predict(self, note_text: str, visit_date: str) -> list:
        # 1. Run windowed NER
        act_spans_char, tim_spans_char = self.predict_ner_spans_windowed(note_text)
        
        # 2. Run windowed Biaffine Linking
        pred_act_keep, pred_tim_keep, pred_links = self.predict_links_for_predicted_spans_windowed(
            note_text, act_spans_char, tim_spans_char
        )

        standardized_outputs = []
        for i, (as0, ae0, raw_act) in enumerate(pred_act_keep):
            j = pred_links[i] if i < len(pred_links) else None
            
            # Action Normalization (Entity correction for partial spans)
            canonical_action = normalize_action_with_expansion(note_text, as0, ae0)
            if canonical_action not in ACTION_ONTOLOGY:
                continue

            if j is not None and j < len(pred_tim_keep):
                ts0, te0, raw_time = pred_tim_keep[j]
                
                # Deterministic Time Normalization
                days = time_text_to_days_offset(raw_time)
                if days is not None:
                    standardized_outputs.append({
                        "action": canonical_action,
                        "days_offset": int(days)
                    })
                    
        return standardized_outputs
SYSTEM_LLAMA = (
    "You are an expert clinical information extraction system. "
    "Extract scheduled follow-up actions and their timing from the note."
)

class LlamaExtractor(ClinicalExtractor):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _extract_json_array(self, text: str):
        """Robustly parses a JSON array from the generated text string."""
        if not text:
            return []
        i = text.find("[")
        j = text.rfind("]")
        if i == -1 or j == -1 or j < i:
            return []
        cand = text[i:j+1].strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            return []

    def build_prompt(self, note_text: str) -> str:
        """Constructs the prompt dynamically using the ACTION_ONTOLOGY."""
        # 1. Pull the canonical closed set dynamically
        allowed_actions = list(ACTION_ONTOLOGY.keys())
        actions_list = "\n".join([f"- {a}" for a in allowed_actions])
        
        user_prompt = f"""Task:
From the clinical note, extract ONLY scheduled follow-up items that match the allowed closed-set actions.
Return a JSON array (possibly empty). Each element MUST have exactly these keys:
- action (one of the allowed actions, exact string)
- period_date (YYYY-MM-DD computed relative to the visit date)

Rules:
- Use ONLY actions from the allowed list below (no new actions).
- period_date must be computed using the visit date in the note.
- If no scheduled items: return [].
- Output JSON only (no markdown, no extra text).

Allowed actions (closed set):
{actions_list}

Clinical note:
{note_text}
"""
        messages = [
            {"role": "system", "content": SYSTEM_LLAMA},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def predict(self, note_text: str, visit_date: str) -> list:
        # 1. Generate the Prompt
        prompt = self.build_prompt(note_text)

        # 2. Run Inference
        # Uses the model's current device automatically
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            stop_strings=["<|eot_id|>"],
            tokenizer=self.tokenizer,
        )
        
        # 3. Decode & Extract JSON
        gen = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        parsed_list = self._extract_json_array(gen)

        # 4. Standardize the Output Format
        standardized_outputs = []
        
        # Parse the note's visit_date to act as the mathematical anchor
        try:
            v_date = datetime.strptime(visit_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            v_date = None

        for item in parsed_list:
            if not isinstance(item, dict):
                continue
                
            # Filter via Ontology
            canonical_action = normalize_action(item.get("action", ""))
            if canonical_action not in ACTION_ONTOLOGY:
                continue
                
            # Date Arithmetic (period_date - visit_date = days_offset)
            days = None
            p_date_str = item.get("period_date", "")
            if p_date_str and v_date:
                try:
                    p_date = datetime.strptime(p_date_str, "%Y-%m-%d")
                    days = (p_date - v_date).days
                except (ValueError, TypeError):
                    pass
            
            # Append if valid
            if days is not None:
                standardized_outputs.append({
                    "action": canonical_action,
                    "days_offset": int(days)
                })
                
        return standardized_outputs

class ChatGPTExtractor(ClinicalExtractor):
    def __init__(self, async_client, model_name="gpt-4o-mini"):
        self.client = async_client
        self.model_name = model_name

    def _extract_list_from_json_obj(self, data):
        """
        Handles OpenAI's json_object response format. 
        Sometimes it returns a dict like {"items": [...]}, other times a direct list.
        """
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return v
            return []
        return []

    def build_prompt(self, note_text: str, visit_date: str) -> str:
        """Constructs the prompt dynamically using the ACTION_ONTOLOGY."""
        allowed_actions = list(ACTION_ONTOLOGY.keys())
        actions_list = "\n".join([f"- {a}" for a in allowed_actions])
        
        return f"""Task:
Extract ONLY scheduled follow-up items from the clinical note that match the allowed closed-set actions below.
Return a JSON array of objects with keys: "action", "period_date".

Rules:
1. "action": Must be an EXACT string match from the Allowed Actions list.
2. "period_date": Calculate the YYYY-MM-DD date based on the Visit Date ({visit_date}) and the time phrase in the note.
3. Ignore history, past tests, or items not in the allowed list.
4. If no items found, return [].

Allowed Actions:
{actions_list}

Clinical Note:
{note_text}
"""

    async def predict_async(self, note_text: str, visit_date: str) -> list:
        prompt = self.build_prompt(note_text, visit_date)
        
        try:
            # 1. Async API Call
            resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful clinical assistant. Output only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            
            # 2. Extract JSON payload
            raw_content = resp.choices[0].message.content
            data = json.loads(raw_content)
            parsed_list = self._extract_list_from_json_obj(data)
            
        except Exception as e:
            print(f"API Error on note: {e}")
            parsed_list = []

        # 3. Standardize Outputs (Identical to LlamaExtractor)
        standardized_outputs = []
        
        try:
            v_date = datetime.strptime(visit_date, "%Y-%m-%d")
        except (ValueError, TypeError):
            v_date = None

        for item in parsed_list:
            if not isinstance(item, dict):
                continue
                
            # Filter via Ontology
            canonical_action = normalize_action(item.get("action", ""))
            if canonical_action not in ACTION_ONTOLOGY:
                continue
                
            # Date Arithmetic (period_date - visit_date = days_offset)
            days = None
            p_date_str = item.get("period_date", "")
            if p_date_str and v_date:
                try:
                    p_date = datetime.strptime(p_date_str, "%Y-%m-%d")
                    days = (p_date - v_date).days
                except (ValueError, TypeError):
                    pass
            
            # Append if valid
            if days is not None:
                standardized_outputs.append({
                    "action": canonical_action,
                    "days_offset": int(days)
                })
                
        return standardized_outputs