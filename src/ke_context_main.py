import argparse
import os

import dill
import numpy as np
import torch

from modules.KEContextMed import KnowledgeEnhancedContextMed
from training_ke_context import TestKEContext, TrainKEContext


def set_seed():
    torch.manual_seed(1203)
    np.random.seed(2048)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1203)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="run a very small CPU experiment")
    parser.add_argument("--Test", action="store_true", help="test mode")
    parser.add_argument("--dataset", default="mimic3", help="mimic3/mimic4")
    parser.add_argument("--run_name", default="ke_context", help="subdirectory under saved/{dataset}")
    parser.add_argument("--resume_path", default=None, type=str, help="model checkpoint for test mode")
    parser.add_argument("--device", type=int, default=0, help="gpu id, negative for cpu")

    parser.add_argument("--dim", default=128, type=int, help="model dimension")
    parser.add_argument("--batch_size", default=1, type=int, help="kept for compatibility")
    parser.add_argument("--lr", default=5e-4, type=float, help="learning rate")
    parser.add_argument("--dp", default=0.5, type=float, help="dropout ratio")
    parser.add_argument("--target_ddi", type=float, default=0.06, help="expected ddi for training")
    parser.add_argument("--coef", default=2.5, type=float, help="coefficient for DDI loss annealing")
    parser.add_argument("--epochs", default=25, type=int, help="training epochs")
    parser.add_argument(
        "--finetune_semantic",
        action="store_true",
        help="fine-tune BioBERT semantic embeddings instead of freezing them",
    )

    args = parser.parse_args()
    if args.Test and args.resume_path is None:
        raise FileNotFoundError("Test mode needs --resume_path")
    return args


def load_pickle(path):
    with open(path, "rb") as f:
        return dill.load(f)


def load_optional_embedding(path, device):
    if not os.path.exists(path):
        print(f"Warning: semantic embedding not found: {path}")
        return None
    print(f"Loading semantic embedding: {path}")
    return torch.load(path, map_location=device)


def to_matrix(value):
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return np.asarray(value)


def to_torch_float(value, device):
    if torch.is_tensor(value):
        return value.float().to(device)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return torch.tensor(value, dtype=torch.float32, device=device)


if __name__ == "__main__":
    project_src = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_src)

    set_seed()
    args = parse_args()
    project_root = os.path.dirname(project_src)
    args.output_dir = os.path.join(project_root, "saved_ke_context", args.dataset, args.run_name)
    print(args)

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 and not args.debug else "cpu"
    )

    data_dir = os.path.join(project_root, "data", args.dataset)
    data_path = os.path.join(data_dir, "output", "records_final.pkl")
    voc_path = os.path.join(data_dir, "output", "voc_final.pkl")
    ddi_adj_path = os.path.join(data_dir, "output", "ddi_A_final.pkl")
    diag_prior_path = os.path.join(data_dir, "graphs", "Diag_Med_relevance.pkl")
    proc_prior_path = os.path.join(data_dir, "graphs", "Proc_Med_relevance.pkl")

    diag_emb_path = os.path.join(data_dir, "embeddings", "biobert_diag_emb.pt")
    proc_emb_path = os.path.join(data_dir, "embeddings", "biobert_proc_emb.pt")
    med_emb_path = os.path.join(data_dir, "embeddings", "biobert_med_emb.pt")

    ddi_adj = to_torch_float(load_pickle(ddi_adj_path), device)

    data = load_pickle(data_path)
    adm_id = 0
    for patient in data:
        for adm in patient:
            if len(adm) == 3:
                adm.append(adm_id)
            else:
                adm[3] = adm_id
            adm_id += 1
    if args.debug:
        data = data[:5]

    voc = load_pickle(voc_path)
    diag_voc, pro_voc, med_voc = voc["diag_voc"], voc["pro_voc"], voc["med_voc"]
    voc_size = [
        len(diag_voc.idx2word),
        len(pro_voc.idx2word),
        len(med_voc.idx2word),
    ]

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point + eval_len:]

    structural_prior_diag_med = to_matrix(load_pickle(diag_prior_path))
    structural_prior_proc_med = to_matrix(load_pickle(proc_prior_path))

    pretrained_diag_emb = load_optional_embedding(diag_emb_path, device)
    pretrained_proc_emb = load_optional_embedding(proc_emb_path, device)
    pretrained_med_emb = load_optional_embedding(med_emb_path, device)

    model = KnowledgeEnhancedContextMed(
        structural_prior_diag_med=structural_prior_diag_med,
        structural_prior_proc_med=structural_prior_proc_med,
        tensor_ddi_adj=ddi_adj,
        voc_size=voc_size,
        emb_dim=args.dim,
        dropout=args.dp,
        device=device,
        pretrained_diag_emb=pretrained_diag_emb,
        pretrained_proc_emb=pretrained_proc_emb,
        pretrained_med_emb=pretrained_med_emb,
        freeze_semantic=not args.finetune_semantic,
    ).to(device)

    print("Leakage note: drug semantic embeddings are used as static external knowledge.")
    print("They are safe if generated from drug text/ATC descriptions, not from full prescription labels.")
    print("1.Training Phase")

    if args.Test:
        print("Test mode, skip training phase")
        model.load_state_dict(torch.load(args.resume_path, map_location=device))
    else:
        model = TrainKEContext(model, device, data_train, data_eval, voc_size, args)

    print("2.Testing Phase")
    TestKEContext(model, device, data_test, voc_size)
