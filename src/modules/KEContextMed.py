import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_tensor(value, dtype=torch.float32, device=None):
    if torch.is_tensor(value):
        return value.detach().clone().to(dtype=dtype, device=device)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return torch.tensor(value, dtype=dtype, device=device)


class ContextBlock(nn.Module):
    def __init__(self, emb_dim, num_heads=4, dropout=0.2):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        if x.size(1) == 0:
            return x
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))


class StructuralPriorInjector(nn.Module):
    def __init__(self, emb_dim, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, entity_emb, prior_rows, drug_semantics):
        if entity_emb.size(0) == 0:
            return entity_emb

        prior_rows = prior_rows.clamp_min(0)
        row_sum = prior_rows.sum(dim=-1, keepdim=True)
        prior_norm = prior_rows / row_sum.clamp_min(1e-8)
        prior_norm = torch.where(row_sum > 0, prior_norm, torch.zeros_like(prior_norm))

        knowledge_context = prior_norm.matmul(drug_semantics)
        gate = self.gate(torch.cat([entity_emb, knowledge_context], dim=-1))
        return self.norm(entity_emb + gate * knowledge_context)


class StructuralPriorCalibrator(nn.Module):
    def __init__(self, emb_dim, num_med):
        super().__init__()
        self.gate = nn.Linear(emb_dim, num_med)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.norm = nn.LayerNorm(num_med)

    def forward(self, logits, patient_context, structural_prior):
        structural_prior = structural_prior.clamp_min(0)
        prior_sum = structural_prior.sum(dim=-1, keepdim=True)
        prior = structural_prior / prior_sum.clamp_min(1e-8)
        prior = torch.where(prior_sum > 0, prior, torch.zeros_like(prior))
        prior_signal = self.norm(prior)
        gate = torch.sigmoid(self.gate(patient_context))
        return logits + self.scale * gate * prior_signal


class KnowledgeEnhancedContextMed(nn.Module):
    """
    Knowledge-enhanced context model for medication recommendation.

    Knowledge means structural priors between clinical entities and medications.
    The model intentionally removes molecule/SMILES embedding paths. Medication
    representations come from drug semantic embeddings when available, or from a
    trainable medication table otherwise.
    """

    def __init__(
        self,
        structural_prior_diag_med,
        structural_prior_proc_med,
        tensor_ddi_adj,
        voc_size,
        emb_dim=128,
        dropout=0.5,
        device=torch.device("cpu"),
        pretrained_diag_emb=None,
        pretrained_proc_emb=None,
        pretrained_med_emb=None,
        freeze_semantic=True,
    ):
        super().__init__()
        self.device = device
        self.emb_dim = emb_dim
        self.num_diag, self.num_proc, self.num_med = voc_size[:3]

        self.diag_embedding, self.diag_proj = self._build_embedding(
            self.num_diag, emb_dim, pretrained_diag_emb, freeze_semantic
        )
        self.proc_embedding, self.proc_proj = self._build_embedding(
            self.num_proc, emb_dim, pretrained_proc_emb, freeze_semantic
        )
        self.med_embedding, self.med_proj = self._build_embedding(
            self.num_med, emb_dim, pretrained_med_emb, freeze_semantic
        )

        self.register_buffer("structural_prior_diag_med", safe_tensor(structural_prior_diag_med))
        self.register_buffer("structural_prior_proc_med", safe_tensor(structural_prior_proc_med))
        self.register_buffer("tensor_ddi_adj", tensor_ddi_adj.float())

        self.diag_context = ContextBlock(emb_dim, dropout=dropout)
        self.proc_context = ContextBlock(emb_dim, dropout=dropout)
        self.visit_context = ContextBlock(emb_dim, dropout=dropout)
        self.temporal_context = ContextBlock(emb_dim, dropout=dropout)

        self.diag_prior_injector = StructuralPriorInjector(emb_dim, dropout)
        self.proc_prior_injector = StructuralPriorInjector(emb_dim, dropout)
        self.calibrator = StructuralPriorCalibrator(emb_dim, self.num_med)

        self.token_type = nn.Parameter(torch.randn(4, emb_dim) * 0.02)
        self.output = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, self.num_med),
        )

        self.init_weights()

    def _build_embedding(self, num_items, emb_dim, pretrained_emb, freeze):
        if pretrained_emb is None:
            embedding = nn.Embedding(num_items, emb_dim)
            projection = nn.Identity()
            return embedding, projection

        pretrained_emb = safe_tensor(pretrained_emb)
        embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=freeze)
        if pretrained_emb.size(1) == emb_dim:
            projection = nn.Identity()
        else:
            projection = nn.Linear(pretrained_emb.size(1), emb_dim)
        return embedding, projection

    def init_weights(self):
        initrange = 0.1
        for embedding in [self.diag_embedding, self.proc_embedding, self.med_embedding]:
            if embedding.weight.requires_grad:
                embedding.weight.data.uniform_(-initrange, initrange)

    def _drug_semantics(self):
        return self.med_proj(self.med_embedding.weight)

    def _embed_diag(self, diag_ids):
        return self.diag_proj(self.diag_embedding(diag_ids))

    def _embed_proc(self, proc_ids):
        return self.proc_proj(self.proc_embedding(proc_ids))

    def _embed_med(self, med_ids):
        return self.med_proj(self.med_embedding(med_ids))

    def _pool(self, x):
        if x.size(0) == 0:
            return torch.zeros(self.emb_dim, device=self.device)
        return x.mean(dim=0)

    def _prior_vector(self, diag_ids, proc_ids):
        pieces = []
        if diag_ids.numel() > 0:
            pieces.append(self.structural_prior_diag_med[diag_ids].mean(dim=0))
        if proc_ids.numel() > 0:
            pieces.append(self.structural_prior_proc_med[proc_ids].mean(dim=0))
        if not pieces:
            return torch.zeros(self.num_med, device=self.device)
        return torch.stack(pieces, dim=0).mean(dim=0)

    def _visit_representation(self, adm, previous_meds):
        diag_ids = torch.LongTensor(adm[0]).to(self.device)
        proc_ids = torch.LongTensor(adm[1]).to(self.device)
        drug_semantics = self._drug_semantics()

        diag_emb = self._embed_diag(diag_ids)
        proc_emb = self._embed_proc(proc_ids)

        if diag_ids.numel() > 0:
            diag_prior = self.structural_prior_diag_med[diag_ids]
            diag_emb = self.diag_prior_injector(diag_emb, diag_prior, drug_semantics)
            diag_emb = self.diag_context(diag_emb.unsqueeze(0)).squeeze(0)

        if proc_ids.numel() > 0:
            proc_prior = self.structural_prior_proc_med[proc_ids]
            proc_emb = self.proc_prior_injector(proc_emb, proc_prior, drug_semantics)
            proc_emb = self.proc_context(proc_emb.unsqueeze(0)).squeeze(0)

        if previous_meds:
            prev_med_ids = torch.LongTensor(previous_meds).to(self.device)
            prev_med_rep = self._pool(self._embed_med(prev_med_ids))
        else:
            prev_med_rep = torch.zeros(self.emb_dim, device=self.device)

        prior_vector = self._prior_vector(diag_ids, proc_ids)
        prior_sum = prior_vector.sum()
        if prior_sum > 0:
            structural_rep = (prior_vector / prior_sum.clamp_min(1e-8)).matmul(drug_semantics)
        else:
            structural_rep = torch.zeros(self.emb_dim, device=self.device)

        tokens = torch.stack(
            [
                self._pool(diag_emb),
                self._pool(proc_emb),
                prev_med_rep,
                structural_rep,
            ],
            dim=0,
        )
        tokens = tokens + self.token_type
        tokens = self.visit_context(tokens.unsqueeze(0)).squeeze(0)
        return tokens.mean(dim=0), prior_vector

    def forward(self, patient_data, calc_cf=True):
        visit_reps = []
        current_prior = None

        for adm_idx, adm in enumerate(patient_data):
            previous_meds = patient_data[adm_idx - 1][2] if adm_idx > 0 else []
            visit_rep, current_prior = self._visit_representation(adm, previous_meds)
            visit_reps.append(visit_rep)

        sequence = torch.stack(visit_reps, dim=0).unsqueeze(0)
        sequence = self.temporal_context(sequence).squeeze(0)
        patient_context = sequence[-1:].contiguous()

        logits = self.output(patient_context)
        logits = self.calibrator(logits, patient_context, current_prior.unsqueeze(0))

        pred_prob = torch.sigmoid(logits)
        ddi_prob = torch.matmul(pred_prob.t(), pred_prob)
        ddi_loss = 0.0005 * ddi_prob.mul(self.tensor_ddi_adj).sum()

        return logits, ddi_loss
