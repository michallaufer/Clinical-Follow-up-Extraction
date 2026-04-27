import re
import torch

# Editable ontology: canonical -> list of aliases
ACTION_ONTOLOGY = {
  "CT Scan": ["ct scan", "ct test", "ct lab", "computed tomography", "ct"],
  "MRI": ["mri", "magnetic resonance imaging"],
  "MRI Brain": ["mri brain", "brain mri"],
  "X-Ray": ["x-ray", "xray", "x ray", "radiograph", "xr"],
  "Blood Test": ["blood test", "labs", "lab test", "bloodwork", "cbc", "cmp"],
  "Physical Therapy": ["physical therapy", "pt", "physio", "physiotherapy"],
  "Echocardiogram": ["echocardiogram", "echo", "tte"],
  "Stress Test": ["stress test", "treadmill test"],
  "Holter Monitor": ["holter", "holter monitor"],
  "Pulmonary Function Test": ["pft", "pulmonary function test"],
  "Cardiac MRI": ["cardiac mri", "mri heart"],
  "Cardiology Consult": ["cardiology consult", "cardiology referral"],
  "Endoscopy": ["endoscopy", "egd"],
  "Colonoscopy": ["colonoscopy", "colon scope"],
  "Stool Antigen Test": ["stool antigen test", "stool test"],
  "Abdominal Ultrasound": ["abdominal ultrasound", "abd us", "us abdomen"],
  "GI Consult": ["gi consult", "gastroenterology consult", "gi referral"],
  "Breath Test": ["breath test"],
  "EEG": ["eeg", "electroencephalogram"],
  "EMG": ["emg", "electromyography"],
  "Neurology Consult": ["neurology consult", "neurology referral"],
  "Sleep Study": ["sleep study", "polysomnography"],
  "Orthopedic Consult": ["orthopedic consult", "ortho consult", "orthopedics referral"],
  "Joint Injection": ["joint injection", "steroid injection"],
  "Annual Physical": ["annual physical", "yearly physical"],
  "Vaccination": ["vaccination", "vaccine", "immunization", "vac", "vax", "shot"],
  "Urinalysis": ["urinalysis", "urine test", "ua"],
  "Lipid Panel": ["lipid panel", "cholesterol test"],
}

def _norm(s: str) -> str:
    s = (s or "").casefold().strip()
    s = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2212\-]+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def build_alias_index(ontology: dict) -> dict:
    idx = {}
    for canon, aliases in ontology.items():
        idx[_norm(canon)] = canon
        for al in aliases:
            idx[_norm(al)] = canon
    return idx

ALIAS_INDEX = build_alias_index(ACTION_ONTOLOGY)

def normalize_action(surface_text: str) -> str:
    key = _norm(surface_text)
    return ALIAS_INDEX.get(key, key)

def normalize_action_with_expansion(note: str, as0: int, ae0: int) -> str:
    surface = note[as0:ae0]
    canon = normalize_action(surface)
    if canon in ACTION_ONTOLOGY:
        return canon

    if len(surface.strip()) <= 10:
        end = min(len(note), as0 + 60)
        chunk = note[as0:end]
        stop = re.search(r"[.;:\n]", chunk)
        if stop:
            chunk = chunk[:stop.start()]

        words = chunk.strip().split()
        for k in range(1, min(6, len(words)) + 1):
            cand = " ".join(words[:k]).strip()
            cand_canon = normalize_action(cand)
            if cand_canon in ACTION_ONTOLOGY:
                return cand_canon

        if words:
            cand_canon = normalize_action(words[0])
            if cand_canon in ACTION_ONTOLOGY:
                return cand_canon

    return canon

@torch.no_grad()
def normalize_test_with_model(surface_text: str, biobert_model, biobert_tokenizer, device, sim_threshold: float = 0.55) -> str:
    """Fallback utilizing BioBERT encoder embeddings if alias lookup fails."""
    key = _norm(surface_text)
    if key in ALIAS_INDEX:
        return ALIAS_INDEX[key]
        
    canon_list = list(ACTION_ONTOLOGY.keys())
    
    def _embed(phrase):
        enc = biobert_tokenizer(phrase, truncation=True, max_length=32, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        H = biobert_model.encoder(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        return ((H * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)).squeeze(0)

    canon_emb = torch.stack([_embed(c).detach().cpu() for c in canon_list], dim=0)
    v = _embed(surface_text).detach().cpu()
    sims = torch.nn.functional.cosine_similarity(canon_emb, v.unsqueeze(0), dim=1)
    
    best = int(torch.argmax(sims).item())
    if float(sims[best].item()) >= sim_threshold:
        return canon_list[best]
        
    return "UNK"