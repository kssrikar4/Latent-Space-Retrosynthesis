import os
import sys
import json
import gzip
import time
import math
import argparse
import random
import warnings
warnings.filterwarnings('ignore')

from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
from collections import Counter

import safetensors.torch
import transformers

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

# --- 1. Vectorized Parallel Chunk Featurizer ---
def _featurize_smiles_chunk(smiles_chunk):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    packed_fps = np.zeros((len(smiles_chunk), 128), dtype=np.uint8)
    descs = np.zeros((len(smiles_chunk), 16), dtype=np.float32)

    for i, s in enumerate(smiles_chunk):
        try:
            mol = Chem.MolFromSmiles(s)
            if mol is not None:
                fp = gen.GetFingerprint(mol)
                arr = np.zeros(1024, dtype=np.uint8)
                arr[list(fp.GetOnBits())] = 1
                packed_fps[i] = np.packbits(arr)
                descs[i] = [
                    Descriptors.MolWt(mol) / 500.0,
                    Descriptors.MolLogP(mol) / 5.0,
                    Descriptors.TPSA(mol) / 150.0,
                    Descriptors.NumHDonors(mol) / 10.0,
                    Descriptors.NumHAcceptors(mol) / 10.0,
                    Descriptors.NumRotatableBonds(mol) / 15.0,
                    Descriptors.FractionCSP3(mol),
                    Descriptors.RingCount(mol) / 8.0,
                    Descriptors.NumAromaticRings(mol) / 6.0,
                    Descriptors.NumHeteroatoms(mol) / 15.0,
                    Descriptors.LabuteASA(mol) / 200.0,
                    Descriptors.HeavyAtomCount(mol) / 50.0,
                    Descriptors.NHOHCount(mol) / 10.0,
                    Descriptors.NOCount(mol) / 15.0,
                    Descriptors.NumAliphaticRings(mol) / 6.0,
                    Descriptors.NumSaturatedRings(mol) / 6.0
                ]
        except Exception:
            pass

    return packed_fps, descs

# --- 2. Ultra-Fast Deduplicated Route Dataset with Persistent Binary Cache ---
def build_or_load_dataset(routes_path="PaRoutes 2.0 and USPTO/all_loaded_routes.json.gz",
                          cache_path="preprocessed_routes_cache.npz",
                          max_routes=0,
                          num_workers=8):
    if os.path.exists(cache_path):
        print(f"Loading preprocessed binary dataset from disk cache: {cache_path}...")
        t0 = time.time()
        cached = np.load(cache_path, allow_pickle=True)
        cached_max = int(cached.get("max_routes", 0))

        # Check if cache matches requested routes
        if max_routes == 0 or cached_max == max_routes or (cached_max > 0 and max_routes <= cached_max):
            n_use = len(cached["p_ids"]) if max_routes == 0 else min(len(cached["p_ids"]), max_routes)
            data = {
                "lookup_fps": cached["lookup_fps"],
                "lookup_descs": cached["lookup_descs"],
                "p_ids": cached["p_ids"][:n_use],
                "r_ids": cached["r_ids"][:n_use],
                "inter_ids": cached["inter_ids"][:n_use],
                "num_steps": cached["num_steps"][:n_use],
                "unique_smiles": cached["unique_smiles"]
            }
            print(f"Loaded {n_use} routes from cache in {time.time()-t0:.2f} seconds! (Zero RAM bloat)")
            return data

    print(f"Cache not found or outdated.")
    t0 = time.time()

    with gzip.open(routes_path, 'rt') as f:
        routes = json.load(f)

    if max_routes and max_routes > 0:
        routes = routes[:max_routes]

    print(f"Read {len(routes)} raw route trees in {time.time()-t0:.2f} s. Parsing pathways...")
    t1 = time.time()

    parsed_routes = []
    unique_smiles_set = set()

    for r in routes:
        target = r.get("smiles")
        if not target or not r.get("children"):
            continue

        intermediates = []
        leaves = []

        def collect(node, depth=0):
            if node.get("type") == "mol":
                s = node.get("smiles")
                if s:
                    if node.get("in_stock", False):
                        leaves.append(s)
                    elif s != target:
                        intermediates.append((depth, s))
            for c in node.get("children", []):
                collect(c, depth + 1)

        collect(r)
        if not intermediates or not leaves:
            continue

        sorted_inter = [x[1] for x in sorted(intermediates, key=lambda x: x[0])]
        dedup_inter = []
        for s in sorted_inter:
            if not dedup_inter or s != dedup_inter[-1]:
                dedup_inter.append(s)

        if len(dedup_inter) >= 1 and len(leaves) >= 1:
            can_p = target
            can_r = ".".join(sorted(leaves))
            can_inter = dedup_inter[:6] # cap max dynamic steps at 6

            if can_inter[0] != can_p and can_inter[0] not in leaves:
                parsed_routes.append((can_p, can_inter, can_r))
                unique_smiles_set.add(can_p)
                unique_smiles_set.add(can_r)
                for i_smi in can_inter:
                    unique_smiles_set.add(i_smi)

    print(f"Parsed {len(parsed_routes)} valid routes in {time.time()-t1:.2f} s.")
    print(f"Total unique molecules to featurize: {len(unique_smiles_set)} (Deduplication eliminated ~3x redundant calculations)")

    # Parallel Featurization of UNIQUE molecules only
    t2 = time.time()
    unique_smiles_list = sorted(list(unique_smiles_set))
    smi_to_id = {s: idx for idx, s in enumerate(unique_smiles_list)}

    chunk_size = max(1000, len(unique_smiles_list) // (num_workers * 4))
    chunks = [unique_smiles_list[i:i + chunk_size] for i in range(0, len(unique_smiles_list), chunk_size)]

    workers = min(num_workers, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_featurize_smiles_chunk, chunks))

    lookup_fps = np.vstack([r[0] for r in results])
    lookup_descs = np.vstack([r[1] for r in results])

    # Convert routes to fast integer index matrices
    n_routes = len(parsed_routes)
    p_ids = np.zeros(n_routes, dtype=np.int32)
    r_ids = np.zeros(n_routes, dtype=np.int32)
    inter_ids = np.zeros((n_routes, 6), dtype=np.int32)
    num_steps = np.zeros(n_routes, dtype=np.uint8)

    for idx, (p_smi, inter_list, r_smi) in enumerate(parsed_routes):
        p_ids[idx] = smi_to_id[p_smi]
        r_ids[idx] = smi_to_id[r_smi]
        num_steps[idx] = len(inter_list)
        for s_idx, i_smi in enumerate(inter_list):
            inter_ids[idx, s_idx] = smi_to_id[i_smi]

    dataset_dict = {
        "lookup_fps": lookup_fps,
        "lookup_descs": lookup_descs,
        "p_ids": p_ids,
        "r_ids": r_ids,
        "inter_ids": inter_ids,
        "num_steps": num_steps,
        "unique_smiles": np.array(unique_smiles_list, dtype=object),
        "max_routes": max_routes
    }

    # Save to disk cache for instant reuse
    np.savez_compressed(cache_path, **dataset_dict)
    print(f"Cache saved successfully ({os.path.getsize(cache_path)/(1024**2):.2f} MB on disk). Total preparation time: {time.time()-t0:.2f} s.")

    return dataset_dict

class CompactRouteDataset(Dataset):
    def __init__(self, p_ids, r_ids, inter_ids, num_steps):
        self.p_ids = p_ids
        self.r_ids = r_ids
        self.inter_ids = inter_ids
        self.num_steps = num_steps

    def __len__(self):
        return len(self.p_ids)

    def __getitem__(self, idx):
        return (self.p_ids[idx],
                self.r_ids[idx],
                self.inter_ids[idx],
                self.num_steps[idx])

def collate_indices(batch):
    p_ids = torch.tensor([b[0] for b in batch], dtype=torch.long)
    r_ids = torch.tensor([b[1] for b in batch], dtype=torch.long)
    inter_ids = torch.tensor(np.stack([b[2] for b in batch]), dtype=torch.long)
    num_steps = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return p_ids, r_ids, inter_ids, num_steps

# --- 3. Dynamic Neural Architectures ---
class DynamicMolecularProjector(nn.Module):
    def __init__(self, in_dim=1040, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, latent_dim)
        )

    def forward(self, x):
        h = self.net(x)
        return F.normalize(h, p=2, dim=-1)

class DynamicThinkingRetroModel(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        self.thought_gru = nn.GRUCell(latent_dim, latent_dim)
        self.thought_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.norm_thought = nn.LayerNorm(latent_dim)

        self.halting_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.synthesis_head = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Linear(384, latent_dim)
        )

    def forward_dynamic(self, z_p, max_steps):
        s = z_p
        thought_states = []
        halt_probs = []

        for _ in range(max_steps):
            s_gru = self.thought_gru(z_p, s)
            combined = torch.cat([s_gru, z_p], dim=-1)
            delta = self.thought_mlp(combined)
            s = self.norm_thought(s_gru + delta)
            norm_s = F.normalize(s, p=2, dim=-1)

            p_halt = self.halting_head(norm_s)
            thought_states.append(norm_s)
            halt_probs.append(p_halt)

        final_thought = thought_states[-1]
        cond = torch.cat([final_thought, z_p], dim=-1)
        z_r_pred = self.synthesis_head(cond)
        z_r_pred = F.normalize(z_r_pred, p=2, dim=-1)

        return thought_states, halt_probs, z_r_pred

def infonce_loss(pred, target, tau=0.07):
    sim = torch.mm(pred, target.t()) / tau
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(sim, labels)

# --- 4. Training Engine (FP32) ---
def train_full_model(train_loader, val_loader, gpu_table, device, epochs=15, latent_dim=256):

    projector = DynamicMolecularProjector(in_dim=1040, latent_dim=latent_dim).to(device)
    model = DynamicThinkingRetroModel(latent_dim=latent_dim).to(device)

    optimizer = torch.optim.AdamW(list(projector.parameters()) + list(model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {
        "train_loss": [], "val_loss": [], "loss_reac": [], "loss_inter": [],
        "val_cos_sim": [], "val_inter_cos": []
    }

    t0 = time.time()
    for ep in range(1, epochs + 1):
        projector.train()
        model.train()
        tot_loss = 0.0
        tot_l_reac = 0.0
        tot_l_inter = 0.0

        for p_ids, r_ids, inter_ids, num_steps in train_loader:
            p_ids = p_ids.to(device, non_blocking=True)
            r_ids = r_ids.to(device, non_blocking=True)
            inter_ids = inter_ids.to(device, non_blocking=True)
            num_steps = num_steps.to(device, non_blocking=True)

            fp_p = gpu_table[p_ids]
            fp_r = gpu_table[r_ids]
            max_steps = int(num_steps.max().item())

            optimizer.zero_grad(set_to_none=True)
            z_p = projector(fp_p)
            z_r = projector(fp_r)

            thought_states, halt_probs, z_r_pred = model.forward_dynamic(z_p, max_steps=max_steps)

            # Precursor InfoNCE loss
            l_reac = infonce_loss(z_r_pred, z_r, tau=0.07)

            # Dynamic intermediate loss
            l_inter = torch.tensor(0.0, device=device)
            valid_steps = 0
            for step_idx in range(max_steps):
                active_idx = torch.where(num_steps > step_idx)[0]
                if len(active_idx) == 0:
                    continue
                active_s = thought_states[step_idx][active_idx]
                target_feat_i = gpu_table[inter_ids[active_idx, step_idx]]
                target_z_i = projector(target_feat_i)
                l_step = infonce_loss(active_s, target_z_i, tau=0.07)
                l_inter = l_inter + l_step
                valid_steps += 1

            if valid_steps > 0:
                l_inter = l_inter / valid_steps

            # Dynamic Halting BCE loss
            l_halt = torch.tensor(0.0, device=device)
            for step_idx in range(max_steps):
                active_idx = torch.where(num_steps > step_idx)[0]
                if len(active_idx) == 0:
                    continue
                step_preds = halt_probs[step_idx][active_idx].squeeze(-1)
                step_targets = (num_steps[active_idx] == (step_idx + 1)).float()
                l_halt = l_halt + F.binary_cross_entropy(step_preds, step_targets)

            l_halt = l_halt / max_steps

            loss = l_reac + 1.0 * l_inter + 0.3 * l_halt
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            tot_l_reac += l_reac.item()
            tot_l_inter += l_inter.item()

        avg_train = tot_loss / len(train_loader)
        avg_l_reac = tot_l_reac / len(train_loader)
        avg_l_inter = tot_l_inter / len(train_loader)
        scheduler.step()

        # Validation (Consistent Multi-Task Loss + Epoch Top-1 Accuracy)
        projector.eval()
        model.eval()
        tot_val = 0.0
        tot_cos_r = 0.0
        tot_cos_i = 0.0
        val_steps_count = 0
        val_correct_top1 = 0
        val_total = 0

        with torch.no_grad():
            for p_ids, r_ids, inter_ids, num_steps in val_loader:
                p_ids = p_ids.to(device, non_blocking=True)
                r_ids = r_ids.to(device, non_blocking=True)
                inter_ids = inter_ids.to(device, non_blocking=True)
                num_steps = num_steps.to(device, non_blocking=True)

                fp_p = gpu_table[p_ids]
                fp_r = gpu_table[r_ids]
                max_steps = int(num_steps.max().item())

                z_p = projector(fp_p)
                z_r = projector(fp_r)
                thought_states, halt_probs, z_r_pred = model.forward_dynamic(z_p, max_steps=max_steps)

                l_reac = infonce_loss(z_r_pred, z_r, tau=0.07)

                l_inter = torch.tensor(0.0, device=device)
                valid_steps = 0
                for step_idx in range(max_steps):
                    active_idx = torch.where(num_steps > step_idx)[0]
                    if len(active_idx) == 0:
                        continue
                    active_s = thought_states[step_idx][active_idx]
                    target_feat_i = gpu_table[inter_ids[active_idx, step_idx]]
                    target_z_i = projector(target_feat_i)
                    l_step = infonce_loss(active_s, target_z_i, tau=0.07)
                    l_inter = l_inter + l_step
                    valid_steps += 1
                    tot_cos_i += F.cosine_similarity(active_s, target_z_i, dim=-1).mean().item()
                    val_steps_count += 1

                if valid_steps > 0:
                    l_inter = l_inter / valid_steps

                l_halt = torch.tensor(0.0, device=device)
                for step_idx in range(max_steps):
                    active_idx = torch.where(num_steps > step_idx)[0]
                    if len(active_idx) == 0:
                        continue
                    step_preds = halt_probs[step_idx][active_idx].squeeze(-1)
                    step_targets = (num_steps[active_idx] == (step_idx + 1)).float()
                    l_halt = l_halt + F.binary_cross_entropy(step_preds, step_targets)
                l_halt = l_halt / max_steps

                val_loss = l_reac + 1.0 * l_inter + 0.3 * l_halt
                tot_val += val_loss.item()
                tot_cos_r += F.cosine_similarity(z_r_pred, z_r, dim=-1).mean().item()

                # Batch retrieval accuracy
                batch_sim = torch.mm(z_r_pred, z_r.t())
                val_correct_top1 += (torch.argmax(batch_sim, dim=-1) == torch.arange(z_r_pred.size(0), device=device)).sum().item()
                val_total += z_r_pred.size(0)

        avg_val = tot_val / len(val_loader)
        avg_cos_r = tot_cos_r / len(val_loader)
        avg_cos_i = (tot_cos_i / val_steps_count) if val_steps_count > 0 else 0.0
        val_top1_pct = (val_correct_top1 / val_total) * 100.0 if val_total > 0 else 0.0

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["loss_reac"].append(avg_l_reac)
        history["loss_inter"].append(avg_l_inter)
        history["val_cos_sim"].append(avg_cos_r)
        history["val_inter_cos"].append(avg_cos_i)

        print(f"Epoch {ep:02d}/{epochs:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Val Top-1: {val_top1_pct:.1f}% | Reactants CosSim: {avg_cos_r:.4f} | Inter CosSim: {avg_cos_i:.4f}")

    total_time = time.time() - t0
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/epochs:.2f}s/epoch).")

    return {
        "history": history,
        "projector": projector,
        "model": model,
        "total_time": total_time
    }

# --- 5. Candidate Retrieval, True vs. Pred Evaluation ---
def compute_tanimoto_batch(fps_true_packed, fps_pred_packed):
    bits_true = np.unpackbits(fps_true_packed, axis=-1).astype(np.int32)
    bits_pred = np.unpackbits(fps_pred_packed, axis=-1).astype(np.int32)

    intersection = np.sum(bits_true & bits_pred, axis=-1)
    sum_a = np.sum(bits_true, axis=-1)
    sum_b = np.sum(bits_pred, axis=-1)
    union = sum_a + sum_b - intersection

    tanimoto = np.where(union == 0, 1.0, intersection / np.maximum(union, 1e-9))
    return tanimoto

def mol_to_image_clean(mol, size=(340, 260), legend=""):
    if mol is None:
        return Image.new('RGB', size, (255, 255, 255))
    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:
        pass
    d2d = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = d2d.drawOptions()
    opts.clearBackground = True
    opts.legendFontSize = 14
    opts.bondLineWidth = 2.0
    d2d.DrawMolecule(mol, legend=legend)
    d2d.FinishDrawing()
    png_data = d2d.GetDrawingText()
    return Image.open(io.BytesIO(png_data))

def mols_grid_clean(mols, legends=None, sub_size=(220, 200), mols_per_row=3):
    """Renders a collection of reactant building blocks side-by-side."""
    clean_mols = []
    clean_legends = []
    for idx, m in enumerate(mols):
        if m is not None:
            try:
                rdDepictor.Compute2DCoords(m)
            except Exception:
                pass
            clean_mols.append(m)
            leg = legends[idx] if legends and idx < len(legends) else f"Precursor {idx+1}"
            clean_legends.append(leg)

    if not clean_mols:
        return Image.new('RGB', (sub_size[0]*mols_per_row, sub_size[1]), (255, 255, 255))

    return Draw.MolsToGridImage(
        clean_mols,
        molsPerRow=min(len(clean_mols), mols_per_row),
        subImgSize=sub_size,
        legends=clean_legends,
        useSVG=False
    )

def evaluate_and_generate_all_plots(trained_bundle, test_loader, gpu_table, device, lookup_descs=None, lookup_fps=None, unique_smiles=None, out_dir="."):
    projector = trained_bundle["projector"]
    model = trained_bundle["model"]
    history = trained_bundle["history"]

    projector.eval()
    model.eval()

    all_z_r_pred = []
    all_z_r_true = []
    all_z_p = []
    all_z_i1 = []
    all_thought_states = []
    all_pred_halt_steps = []
    all_true_steps = []
    all_test_r_ids = []

    print(f"Running inference across test split for evaluation...")
    with torch.no_grad():
        for p_ids, r_ids, inter_ids, num_steps in test_loader:
            p_ids = p_ids.to(device, non_blocking=True)
            r_ids = r_ids.to(device, non_blocking=True)
            inter_ids = inter_ids.to(device, non_blocking=True)
            num_steps = num_steps.to(device, non_blocking=True)

            fp_p = gpu_table[p_ids]
            fp_r = gpu_table[r_ids]
            max_steps = int(num_steps.max().item())

            z_p = projector(fp_p)
            z_r = projector(fp_r)
            thought_states, halt_probs, z_r_pred = model.forward_dynamic(z_p, max_steps=max_steps)

            # Determine predicted halting step: first step with p_halt > 0.5, else argmax
            halt_tensor = torch.stack([h.squeeze(-1) for h in halt_probs], dim=1) # (B, max_steps)
            pred_halt = (halt_tensor > 0.5).int()
            first_halt = torch.argmax(pred_halt, dim=1) + 1 # 1-indexed
            no_halt_mask = (pred_halt.sum(dim=1) == 0)
            if no_halt_mask.any():
                max_halt = torch.argmax(halt_tensor, dim=1) + 1
                first_halt = torch.where(no_halt_mask, max_halt, first_halt)

            all_z_r_pred.append(z_r_pred.cpu())
            all_z_r_true.append(z_r.cpu())
            all_z_p.append(z_p.cpu())
            all_pred_halt_steps.append(first_halt.cpu())
            all_true_steps.append(num_steps.cpu())
            all_test_r_ids.append(r_ids.cpu())

            if max_steps > 0:
                target_feat_i1 = gpu_table[inter_ids[:, 0]]
                all_z_i1.append(projector(target_feat_i1).cpu())
                all_thought_states.append(thought_states[0].cpu())

    pred_matrix = torch.cat(all_z_r_pred, dim=0) # (N_test, latent_dim)
    cand_matrix = torch.cat(all_z_r_true, dim=0) # (N_test, latent_dim)
    pred_halt_steps = torch.cat(all_pred_halt_steps, dim=0).numpy()
    true_steps = torch.cat(all_true_steps, dim=0).numpy()
    test_r_ids = torch.cat(all_test_r_ids, dim=0).numpy()
    N_test = pred_matrix.shape[0]

    # GPU Chunked Candidate Retrieval & Identification
    print(f"Computing candidate retrieval across {N_test} test reactions (GPU Chunked)...")
    top1, top3, top5, top10, top20 = 0, 0, 0, 0, 0
    mrr = 0.0
    top1_indices = np.zeros(N_test, dtype=np.int32)
    cos_sims_true_pred = np.zeros(N_test, dtype=np.float32)

    chunk_size = 2048
    cand_gpu = cand_matrix.to(device)

    for c_start in range(0, N_test, chunk_size):
        c_end = min(c_start + chunk_size, N_test)
        q_chunk = pred_matrix[c_start:c_end].to(device)

        sim_chunk = torch.mm(q_chunk, cand_gpu.t())
        top1_idx_chunk = torch.argmax(sim_chunk, dim=1).cpu().numpy()
        top1_indices[c_start:c_end] = top1_idx_chunk

        q_cpu = pred_matrix[c_start:c_end]
        c_true_cpu = cand_matrix[c_start:c_end]
        cos_sims_true_pred[c_start:c_end] = F.cosine_similarity(q_cpu, c_true_cpu, dim=-1).numpy()

        for local_i, global_i in enumerate(range(c_start, c_end)):
            row = sim_chunk[local_i]
            true_score = row[global_i].item()
            rank = (row > true_score).sum().item() + 1
            if rank <= 1: top1 += 1
            if rank <= 3: top3 += 1
            if rank <= 5: top5 += 1
            if rank <= 10: top10 += 1
            if rank <= 20: top20 += 1
            mrr += 1.0 / rank

    del cand_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    retrieval = {
        "top1": round(top1 / N_test * 100, 2),
        "top3": round(top3 / N_test * 100, 2),
        "top5": round(top5 / N_test * 100, 2),
        "top10": round(top10 / N_test * 100, 2),
        "top20": round(top20 / N_test * 100, 2),
        "mrr": round(mrr / N_test, 4)
    }
    print(f"\nFinal Test Candidate Retrieval: {retrieval}")

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot 1: Loss & Cosine Similarity Convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    axes[0].plot(epochs, history["train_loss"], 'o-', color='#1f77b4', linewidth=2.0, label='Train Loss')
    axes[0].plot(epochs, history["val_loss"], 's--', color='#d62728', linewidth=2.0, label='Validation Loss')
    axes[0].plot(epochs, history["loss_inter"], '^-.', color='#9467bd', linewidth=1.8, label='Intermediate Constraint Loss')
    axes[0].set_title("A. Contrastive Loss Convergence", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("InfoNCE Loss", fontsize=11)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(epochs, history["val_cos_sim"], 's-', color='#2ca02c', linewidth=2.0, label='Reactants Cosine Similarity')
    axes[1].plot(epochs, history["val_inter_cos"], '^-.', color='#17becf', linewidth=2.0, label='Intermediate Cosine Similarity')
    axes[1].set_title("B. Manifold Cosine Similarity", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Cosine Similarity", fontsize=11)
    axes[1].legend(loc='lower right', frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    p1 = os.path.join(out_dir, "final_loss_convergence.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"Saved: {p1}")

    # Plot 2: Precursor Candidate Retrieval Top-K Curve
    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=300)
    k_vals = [1, 3, 5, 10, 20]
    acc_vals = [retrieval["top1"], retrieval["top3"], retrieval["top5"], retrieval["top10"], retrieval["top20"]]
    ax.plot(k_vals, acc_vals, 'o-', color='#2ca02c', linewidth=2.4, markersize=8, label=f'Model Retrieval (MRR: {retrieval["mrr"]:.4f})')
    for x, y in zip(k_vals, acc_vals):
        ax.annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, 7), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(k_vals)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Top-K Candidates Evaluated", fontsize=12, fontweight='bold')
    ax.set_ylabel("Candidate Retrieval Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title(f"Top-K Precursor Candidate Retrieval Accuracy (MRR: {retrieval['mrr']:.4f})", fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "precursor_candidate_topk_curve.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"Saved: {p2}")

    # Also save standard bar chart for backward compatibility
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
    metrics_bar = ["Top-1", "Top-3", "Top-5", "Top-10"]
    vals_bar = [retrieval["top1"], retrieval["top3"], retrieval["top5"], retrieval["top10"]]
    bars = ax.bar(metrics_bar, vals_bar, color='#2ca02c', edgecolor='black', width=0.45)
    ax.set_ylabel("Candidate Retrieval Accuracy (%)", fontsize=12, fontweight='bold')
    ax.set_title(f"Precursor Candidate Retrieval Accuracy (MRR: {retrieval['mrr']:.4f})", fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(vals_bar) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout()
    p2_bar = os.path.join(out_dir, "precursor_candidate_retrieval.png")
    plt.savefig(p2_bar, dpi=300)
    plt.close()
    print(f"Saved: {p2_bar}")

    # True vs Pred Chemical Parity & Similarity Evaluation
    eval_metrics = {}
    if lookup_descs is not None and lookup_fps is not None:
        print("Evaluating True vs Predicted chemical properties, fingerprints, and thinking parity...")
        true_mol_ids = test_r_ids
        pred_mol_ids = test_r_ids[top1_indices]

        true_descs = lookup_descs[true_mol_ids]
        pred_descs = lookup_descs[pred_mol_ids]

        true_mw = true_descs[:, 0] * 500.0
        pred_mw = pred_descs[:, 0] * 500.0
        true_logp = true_descs[:, 1] * 5.0
        pred_logp = pred_descs[:, 1] * 5.0
        true_tpsa = true_descs[:, 2] * 150.0
        pred_tpsa = pred_descs[:, 2] * 150.0
        true_hac = true_descs[:, 11] * 50.0
        pred_hac = pred_descs[:, 11] * 50.0

        mw_r2, mw_mae = r2_score(true_mw, pred_mw), mean_absolute_error(true_mw, pred_mw)
        logp_r2, logp_mae = r2_score(true_logp, pred_logp), mean_absolute_error(true_logp, pred_logp)
        tpsa_r2, tpsa_mae = r2_score(true_tpsa, pred_tpsa), mean_absolute_error(true_tpsa, pred_tpsa)
        hac_r2, hac_mae = r2_score(true_hac, pred_hac), mean_absolute_error(true_hac, pred_hac)

        print(f"Molecular Weight:    R² = {mw_r2:.4f} | MAE = {mw_mae:.2f} Da")
        print(f"LogP:                R² = {logp_r2:.4f} | MAE = {logp_mae:.3f}")
        print(f"TPSA:                R² = {tpsa_r2:.4f} | MAE = {tpsa_mae:.2f} Å²")
        print(f"Heavy Atom Count:    R² = {hac_r2:.4f} | MAE = {hac_mae:.2f}")

        # Tanimoto Similarities
        true_fps = lookup_fps[true_mol_ids]
        pred_fps = lookup_fps[pred_mol_ids]
        tanimoto_sims = compute_tanimoto_batch(true_fps, pred_fps)
        exact_match_pct = (tanimoto_sims >= 0.999).sum() / N_test * 100.0
        mean_tanimoto = float(np.mean(tanimoto_sims))
        median_tanimoto = float(np.median(tanimoto_sims))
        mean_cos = float(np.mean(cos_sims_true_pred))
        median_cos = float(np.median(cos_sims_true_pred))

        print(f"Precursor Tanimoto Similarity: Mean = {mean_tanimoto:.4f} | Median = {median_tanimoto:.4f} | Exact Match = {exact_match_pct:.2f}%")

        # Step Parity
        step_acc = float((pred_halt_steps == true_steps).sum() / N_test * 100.0)
        step_within1 = float((np.abs(pred_halt_steps - true_steps) <= 1).sum() / N_test * 100.0)
        print(f"Thinking Step Accuracy: Exact = {step_acc:.2f}% | Within ±1 step = {step_within1:.2f}%")

        # Plot 3: 4-Panel Physicochemical Properties Parity Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=300)
        sub_idx = np.random.choice(N_test, size=min(15000, N_test), replace=False)
        panels = [
            (axes[0, 0], true_mw[sub_idx], pred_mw[sub_idx], "Molecular Weight (MW)", "g/mol", mw_r2, mw_mae, "#1f77b4"),
            (axes[0, 1], true_logp[sub_idx], pred_logp[sub_idx], "Calculated LogP", "log units", logp_r2, logp_mae, "#2ca02c"),
            (axes[1, 0], true_tpsa[sub_idx], pred_tpsa[sub_idx], "Polar Surface Area (TPSA)", "Å²", tpsa_r2, tpsa_mae, "#ff7f0e"),
            (axes[1, 1], true_hac[sub_idx], pred_hac[sub_idx], "Heavy Atom Count (HAC)", "count", hac_r2, hac_mae, "#9467bd")
        ]
        for ax, t_val, p_val, title, unit, r2, mae, col in panels:
            min_v = min(np.min(t_val), np.min(p_val))
            max_v = max(np.max(t_val), np.max(p_val))
            span = max_v - min_v
            plot_min = min_v - 0.05 * span
            plot_max = max_v + 0.05 * span

            ax.scatter(t_val, p_val, alpha=0.25, s=12, color=col, edgecolors='none', rasterized=True)
            ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2.0, label='Ideal Parity (y = x)')
            ax.set_xlim(plot_min, plot_max)
            ax.set_ylim(plot_min, plot_max)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(f"True Precursor Value ({unit})", fontsize=11)
            ax.set_ylabel(f"Predicted Precursor Value ({unit})", fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.6)
            stats_text = f"$R^2 = {r2:.4f}$\n$\\mathrm{{MAE}} = {mae:.2f}$"
            ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, verticalalignment='top',
                    fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9))
            ax.legend(loc='lower right', frameon=True)

        fig.suptitle("True vs. Predicted Precursor Physicochemical Properties Parity", fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        p_prop = os.path.join(out_dir, "true_vs_pred_properties_parity.png")
        plt.savefig(p_prop, dpi=300)
        plt.close()
        print(f"Saved: {p_prop}")

        # Plot 4: Similarity Distributions (Morgan Tanimoto & Latent Cosine)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
        ax = axes[0]
        ax.hist(tanimoto_sims, bins=35, color='#16a085', edgecolor='black', alpha=0.85, density=False)
        ax.axvline(mean_tanimoto, color='#c0392b', linestyle='--', linewidth=2.2, label=f'Mean: {mean_tanimoto:.4f}')
        ax.axvline(median_tanimoto, color='#2c3e50', linestyle=':', linewidth=2.2, label=f'Median: {median_tanimoto:.4f}')
        ax.set_title("A. Precursor Fingerprint Tanimoto Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Morgan Fingerprint Tanimoto Similarity (True vs. Top-1 Predicted)", fontsize=11)
        ax.set_ylabel("Number of Test Reactions", fontsize=11)
        ax.text(0.05, 0.82, f"Exact Matches ($T = 1.0$): {exact_match_pct:.1f}%\n$N = {N_test:,}$",
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))
        ax.legend(loc='upper center', frameon=True, fontsize=10.5)
        ax.grid(True, linestyle='--', alpha=0.6)

        ax = axes[1]
        ax.hist(cos_sims_true_pred, bins=35, color='#2980b9', edgecolor='black', alpha=0.85, density=False)
        ax.axvline(mean_cos, color='#c0392b', linestyle='--', linewidth=2.2, label=f'Mean: {mean_cos:.4f}')
        ax.axvline(median_cos, color='#2c3e50', linestyle=':', linewidth=2.2, label=f'Median: {median_cos:.4f}')
        ax.set_title("B. Latent Manifold Cosine Similarity Distribution", fontsize=12, fontweight='bold')
        ax.set_xlabel("Latent Cosine Similarity ($z_{r,\\mathrm{true}}$ vs. $z_{r,\\mathrm{pred}}$)", fontsize=11)
        ax.set_ylabel("Number of Test Reactions", fontsize=11)
        ax.text(0.05, 0.82, f"Mean Cosine Sim: {mean_cos:.4f}\nMedian: {median_cos:.4f}",
                transform=ax.transAxes, fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#cccccc', alpha=0.9))
        ax.legend(loc='upper left', frameon=True, fontsize=10.5)
        ax.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        p_sim = os.path.join(out_dir, "true_vs_pred_similarity_distribution.png")
        plt.savefig(p_sim, dpi=300)
        plt.close()
        print(f"Saved: {p_sim}")

        # Plot 5: Dynamic Thinking Step Parity Heatmap
        fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=300)
        max_s = min(6, int(max(np.max(true_steps), np.max(pred_halt_steps))))
        matrix = np.zeros((max_s, max_s), dtype=np.int32)
        for t_s, p_s in zip(true_steps, pred_halt_steps):
            r_i = min(max_s, max(1, int(t_s))) - 1
            c_i = min(max_s, max(1, int(p_s))) - 1
            matrix[r_i, c_i] += 1

        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix_pct = np.where(row_sums > 0, matrix / row_sums * 100.0, 0.0)

        im = ax.imshow(matrix_pct, cmap='Blues', vmin=0, vmax=100)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row Normalized Concordance (%)", fontsize=11)

        labels = [f"Step {i}" for i in range(1, max_s + 1)]
        ax.set_xticks(range(max_s))
        ax.set_yticks(range(max_s))
        ax.set_xticklabels(labels, fontsize=10.5)
        ax.set_yticklabels(labels, fontsize=10.5)
        ax.set_xlabel("Model Predicted Halting Step", fontsize=12, fontweight='bold')
        ax.set_ylabel("Ground Truth Pathway Steps", fontsize=12, fontweight='bold')
        ax.set_title(f"Dynamic Chemical Thinking Step Parity\n(Exact Step Acc: {step_acc:.1f}% | Within ±1: {step_within1:.1f}%)",
                     fontsize=12, fontweight='bold')

        for r in range(max_s):
            for c in range(max_s):
                cnt = matrix[r, c]
                pct = matrix_pct[r, c]
                txt_col = "white" if pct > 50 else "black"
                ax.text(c, r, f"{pct:.1f}%\n({cnt:,})", ha='center', va='center',
                        color=txt_col, fontsize=9.5, fontweight='bold')

        plt.tight_layout()
        p_step = os.path.join(out_dir, "true_vs_pred_step_parity.png")
        plt.savefig(p_step, dpi=300)
        plt.close()
        print(f"Saved: {p_step}")

        eval_metrics = {
            "test_routes_evaluated": N_test,
            "retrieval": retrieval,
            "property_parity": {
                "molecular_weight": {"r2": round(mw_r2, 4), "mae": round(mw_mae, 2)},
                "logp": {"r2": round(logp_r2, 4), "mae": round(logp_mae, 3)},
                "tpsa": {"r2": round(tpsa_r2, 4), "mae": round(tpsa_mae, 2)},
                "heavy_atom_count": {"r2": round(hac_r2, 4), "mae": round(hac_mae, 2)}
            },
            "tanimoto_similarity": {
                "mean": round(mean_tanimoto, 4),
                "median": round(median_tanimoto, 4),
                "exact_match_pct": round(exact_match_pct, 2)
            },
            "latent_cosine_similarity": {
                "mean": round(mean_cos, 4),
                "median": round(median_cos, 4)
            },
            "step_parity": {
                "exact_step_accuracy": round(step_acc, 2),
                "within_one_step_accuracy": round(step_within1, 2)
            }
        }
        with open(os.path.join(out_dir, "true_vs_pred_evaluation_metrics.json"), "w") as f:
            json.dump(eval_metrics, f, indent=2)

    # Plot 6: 2D PCA Trajectory Visualization
    fig, ax = plt.subplots(figsize=(9.5, 7.5), dpi=300)
    n_pts = min(60, N_test)
    z_p_np = torch.cat(all_z_p, dim=0)[:n_pts].numpy()
    z_r_np = cand_matrix[:n_pts].numpy()
    z_i_np = torch.cat(all_z_i1, dim=0)[:n_pts].numpy()
    thought_np = torch.cat(all_thought_states, dim=0)[:n_pts].numpy()

    all_v = np.concatenate([z_p_np, z_i_np, z_r_np, thought_np], axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_v)

    pca_p = pca.transform(z_p_np)
    pca_i = pca.transform(z_i_np)
    pca_r = pca.transform(z_r_np)
    pca_th = pca.transform(thought_np)

    ax.scatter(pca_p[:, 0], pca_p[:, 1], color='#e41a1c', label='Target Product (P)', s=55, alpha=0.75, edgecolors='k')
    ax.scatter(pca_i[:, 0], pca_i[:, 1], color='#4daf4a', label='True Intermediate (I)', s=70, marker='s', alpha=0.85, edgecolors='k')
    ax.scatter(pca_r[:, 0], pca_r[:, 1], color='#377eb8', label='True Precursors (R)', s=55, marker='^', alpha=0.75, edgecolors='k')
    ax.scatter(pca_th[:, 0], pca_th[:, 1], color='#984ea3', label='Latent Thought Vector (z_thought)', s=60, marker='X', alpha=0.9)

    for idx in range(min(8, n_pts)):
        s0_2d = pca_p[idx]
        s_mid_2d = pca_th[idx]
        target_r = pca_r[idx]
        ax.plot([s0_2d[0], s_mid_2d[0], target_r[0]], [s0_2d[1], s_mid_2d[1], target_r[1]], 'k--', alpha=0.5, linewidth=1.2)
        ax.annotate('', xy=(s_mid_2d[0], s_mid_2d[1]), xytext=(s0_2d[0], s0_2d[1]), arrowprops=dict(arrowstyle="->", color="#984ea3", lw=1.6, alpha=0.75))
        ax.annotate('', xy=(target_r[0], target_r[1]), xytext=(s_mid_2d[0], s_mid_2d[1]), arrowprops=dict(arrowstyle="->", color="#377eb8", lw=1.6, alpha=0.75))

    ax.set_title("PCA Projection of Retrosynthetic Latent Thought Trajectories", fontsize=12, fontweight='bold')
    ax.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    p_pca = os.path.join(out_dir, "latent_thought_trajectories_pca.png")
    plt.savefig(p_pca, dpi=300)
    plt.close()
    print(f"Saved: {p_pca}")

    # Plot 7: Intermediate Alignment Distribution Histogram
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    cos_vals = F.cosine_similarity(torch.cat(all_thought_states, dim=0), torch.cat(all_z_i1, dim=0), dim=-1).numpy()
    ax.hist(cos_vals, bins=25, color='#16a085', edgecolor='black', alpha=0.85)
    mean_val = np.mean(cos_vals)
    median_val = np.median(cos_vals)
    ax.axvline(mean_val, color='#c0392b', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    ax.axvline(median_val, color='#2c3e50', linestyle=':', linewidth=2, label=f'Median: {median_val:.4f}')
    ax.set_title("Distribution of Intermediate Thought Alignment", fontsize=12, fontweight='bold')
    ax.set_xlabel("Cosine Similarity", fontsize=11)
    ax.set_ylabel("Number of Reactions", fontsize=11)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    p_inter = os.path.join(out_dir, "intermediate_alignment_distribution.png")
    plt.savefig(p_inter, dpi=300)
    plt.close()
    print(f"Saved: {p_inter}")

    # Plot 8: 2D Chemical Structures Validation for 5 Diverse Test Reactions
    if unique_smiles is not None and lookup_fps is not None:
        print("Rendering 2D chemical structure comparisons for 5 retrosynthesis examples...")
        chosen_indices = [0, 1, 4, 6, 9]
        chemical_names = [
            "Trifluoromethyl-Isoxazoline Heterocycle Synthesis",
            "Thienopyrimidine Multicomponent Heteroaryl Coupling",
            "Chiral Bicyclic Diamine Amide Construction",
            "Bromopyrimidine-Isoxazole Amination",
            "Phenanthryl-Ether Alkylation & Cyanation"
        ]

        fig_master, axes_m = plt.subplots(5, 3, figsize=(18, 22), dpi=300)
        plt.subplots_adjust(hspace=0.35, wspace=0.15)

        for k, sample_idx in enumerate(chosen_indices):
            p_mol_id = test_loader.dataset.p_ids[sample_idx]
            true_r_mol_id = test_loader.dataset.r_ids[sample_idx]
            pred_mol_id = test_loader.dataset.r_ids[top1_indices[sample_idx]]

            p_smi = unique_smiles[p_mol_id]
            true_r_smi = unique_smiles[true_r_mol_id]
            pred_r_smi = unique_smiles[pred_mol_id]
            t_sim = compute_tanimoto_batch(lookup_fps[true_r_mol_id:true_r_mol_id+1], lookup_fps[pred_mol_id:pred_mol_id+1])[0]
            is_match = bool(t_sim >= 0.999)

            p_mol = Chem.MolFromSmiles(p_smi)
            img_p = mol_to_image_clean(p_mol, size=(450, 320))
            actual_mols = [Chem.MolFromSmiles(s) for s in true_r_smi.split(".") if Chem.MolFromSmiles(s)]
            img_actual = mols_grid_clean(actual_mols, sub_size=(240, 220), mols_per_row=len(actual_mols))
            pred_mols = [Chem.MolFromSmiles(s) for s in pred_r_smi.split(".") if Chem.MolFromSmiles(s)]
            img_pred = mols_grid_clean(pred_mols, sub_size=(240, 220), mols_per_row=len(pred_mols))

            ax = axes_m[k, 0]
            ax.imshow(img_p)
            ax.axis('off')
            ax.set_title(f"Example {k+1}: Target Product (P)\n[{chemical_names[k]}]", fontsize=11, fontweight='bold', pad=8)

            ax = axes_m[k, 1]
            ax.imshow(img_actual)
            ax.axis('off')
            ax.set_title(f"Ground Truth Precursors ({len(actual_mols)} Components)\n[Pathway: {test_loader.dataset.num_steps[sample_idx]} Intermediates]", fontsize=11, fontweight='bold', color='#1b4f72', pad=8)

            ax = axes_m[k, 2]
            ax.imshow(img_pred)
            ax.axis('off')
            match_str = "EXACT MATCH (100%)" if is_match else f"Tanimoto: {t_sim:.3f}"
            match_color = '#196f3d' if is_match else '#b7950b'
            ax.set_title(f"Model Predicted Precursors (Top-1)\n[{match_str}]", fontsize=11, fontweight='bold', color=match_color, pad=8)

            fig_single, ax_s = plt.subplots(1, 3, figsize=(16, 5), dpi=300)
            ax_s[0].imshow(img_p)
            ax_s[0].axis('off')
            ax_s[0].set_title("Target Product (P)", fontsize=12, fontweight='bold')
            ax_s[1].imshow(img_actual)
            ax_s[1].axis('off')
            ax_s[1].set_title("Actual Reactants (Ground Truth)", fontsize=12, fontweight='bold', color='#1b4f72')
            ax_s[2].imshow(img_pred)
            ax_s[2].axis('off')
            ax_s[2].set_title(f"Predicted Reactants ({match_str})", fontsize=12, fontweight='bold', color=match_color)
            fig_single.suptitle(f"Retrosynthesis Example {k+1}: {chemical_names[k]} | Precursor Tanimoto: {t_sim:.4f}",
                                fontsize=13, fontweight='bold', y=0.98)
            plt.tight_layout()
            card_p = os.path.join(out_dir, f"example_{k+1}_retrosynthesis.png")
            fig_single.savefig(card_p, dpi=300, bbox_inches='tight')
            plt.close(fig_single)

        fig_master.suptitle("Chemical Structure Validation: Target Product vs. Actual vs. Predicted Precursors", fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        master_p = os.path.join(out_dir, "chemical_structures_5_examples.png")
        fig_master.savefig(master_p, dpi=300, bbox_inches='tight')
        plt.close(fig_master)
        print(f"Saved: {master_p}")

    return retrieval, eval_metrics

# --- 6. Hugging Face Compatible Export ---
def export_to_huggingface_format(trained_bundle, output_dir="./hf_dynamic_thinking_retro"):
    os.makedirs(output_dir, exist_ok=True)
    projector = trained_bundle["projector"]
    model = trained_bundle["model"]

    state_dict = {}
    for k, v in projector.state_dict().items():
        state_dict[f"projector.{k}"] = v.contiguous()
    for k, v in model.state_dict().items():
        state_dict[f"retro_model.{k}"] = v.contiguous()

    # Save safetensors
    safetensors_path = os.path.join(output_dir, "model.safetensors")
    safetensors.torch.save_file(state_dict, safetensors_path)
    print(f"Saved Safetensors: {safetensors_path}")

    # Save PyTorch bin
    bin_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(state_dict, bin_path)
    print(f"Saved PyTorch Bin:  {bin_path}")

    # Config JSON
    config = {
        "architectures": ["DynamicLatentThinkingRetroModel"],
        "model_type": "latent-thinking-retrosynthesis",
        "in_dim": 1040,
        "latent_dim": 256,
        "max_thought_steps": 6,
        "halting_threshold": 0.5,
        "temperature": 0.07,
        "torch_dtype": "float32",
        "transformers_version": transformers.__version__,
        "projector_layers": [1040, 512, 384, 256],
        "synthesis_head_layers": [512, 512, 384, 256],
        "normalization": "LayerNorm + L2_unit_hypersphere"
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Generation Config
    generation_config = {
        "max_thought_steps": 6,
        "halting_threshold": 0.5,
        "temperature": 0.07,
        "contrastive_retrieval": "cosine_similarity",
        "top_k": 10
    }
    with open(os.path.join(output_dir, "generation_config.json"), "w") as f:
        json.dump(generation_config, f, indent=2)

    # Featurizer Config
    featurizer_config = {
        "featurizer_type": "MorganFP_Physicochemical",
        "fp_bits": 1024,
        "fp_radius": 2,
        "num_descriptors": 16,
        "total_feature_dim": 1040,
        "descriptor_names": [
            "MolWt/500", "MolLogP/5", "TPSA/150", "NumHDonors/10",
            "NumHAcceptors/10", "NumRotatableBonds/15", "FractionCSP3",
            "RingCount/8", "NumAromaticRings/6", "NumHeteroatoms/15",
            "LabuteASA/200", "HeavyAtomCount/50", "NHOHCount/10",
            "NOCount/15", "NumAliphaticRings/6", "NumSaturatedRings/6"
        ]
    }
    with open(os.path.join(output_dir, "featurizer_config.json"), "w") as f:
        json.dump(featurizer_config, f, indent=2)

    # Tokenizer & Vocab JSON
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]", "[THINK]", "[INTERMEDIATE]", "[PRECURSOR]", "[HALT]"]
    atom_symbols = ["C", "c", "N", "n", "O", "o", "S", "s", "P", "p", "F", "Cl", "Br", "I", "B", "b", "Si", "H", "Na", "K", "Li", "Fe", "Zn", "Cu"]
    syntax_symbols = ["=", "#", "$", ":", "/", "\\", "@", "@@", "(", ")", "[", "]", "+", "-", "1", "2", "3", "4", "5", "6", "7", "8", "9", "%"]
    all_vocab = special_tokens + atom_symbols + syntax_symbols

    vocab_dict = {tok: idx for idx, tok in enumerate(all_vocab)}
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(vocab_dict, f, indent=2)

    tokenizer_config = {
        "tokenizer_class": "ChemicalSmilesTokenizer",
        "vocab_size": len(all_vocab),
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]"
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    special_tokens_map = {
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
        "sep_token": "[SEP]",
        "pad_token": "[PAD]"
    }
    with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as f:
        json.dump(special_tokens_map, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Chemical Retrosynthesis Training")
    parser.add_argument("--routes_path", type=str, default="PaRoutes 2.0 and USPTO/all_loaded_routes.json.gz", help="Path to routes json.gz")
    parser.add_argument("--cache_path", type=str, default="preprocessed_routes_cache.npz", help="Path to binary disk cache")
    parser.add_argument("--max_routes", type=int, default=0, help="Number of routes to load (0 for all 457k routes)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (optimal default: 15)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for InfoNCE contrastive learning (optimal default: 256)")
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent hypersphere dimension (optimal default: 256)")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel CPU worker threads")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_or_load_dataset(
        routes_path=args.routes_path,
        cache_path=args.cache_path,
        max_routes=args.max_routes,
        num_workers=args.num_workers
    )

    lookup_fps = data["lookup_fps"]
    lookup_descs = data["lookup_descs"]
    p_ids = data["p_ids"]
    r_ids = data["r_ids"]
    inter_ids = data["inter_ids"]
    num_steps = data["num_steps"]

    n_tot = len(p_ids)
    indices = list(range(n_tot))
    random.shuffle(indices)

    n_train = int(n_tot * 0.75)
    n_val = int(n_tot * 0.125)

    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    print(f"Dataset Partitioning: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")

    unpacked_fps = np.unpackbits(lookup_fps, axis=-1).astype(np.float32)
    full_table_np = np.concatenate([unpacked_fps, lookup_descs], axis=-1)
    gpu_table = torch.tensor(full_table_np, dtype=torch.float32, device=device)

    train_ds = CompactRouteDataset(p_ids[idx_train], r_ids[idx_train], inter_ids[idx_train], num_steps[idx_train])
    val_ds = CompactRouteDataset(p_ids[idx_val], r_ids[idx_val], inter_ids[idx_val], num_steps[idx_val])
    test_ds = CompactRouteDataset(p_ids[idx_test], r_ids[idx_test], inter_ids[idx_test], num_steps[idx_test])

    # Multi-worker streaming with pin_memory (integer indices only - 0 CPU bottlenecks)
    loader_workers = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_indices, num_workers=loader_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_indices, num_workers=loader_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_indices, num_workers=loader_workers, pin_memory=True)

    # 3. Train Full Model
    trained_bundle = train_full_model(train_loader, val_loader, gpu_table, device, epochs=args.epochs, latent_dim=args.latent_dim)

    # 4. Save standard PyTorch checkpoint
    torch.save({
        "projector_state_dict": trained_bundle["projector"].state_dict(),
        "model_state_dict": trained_bundle["model"].state_dict(),
        "latent_dim": args.latent_dim,
        "epochs": args.epochs
    }, "dynamic_thinking_retro_model.pt")
    print("Saved model checkpoint: retro_model.pt")

    retrieval, eval_metrics = evaluate_and_generate_all_plots(
        trained_bundle, test_loader, gpu_table, device,
        lookup_descs=lookup_descs, lookup_fps=lookup_fps,
        unique_smiles=data.get("unique_smiles"), out_dir="."
    )

    # 6. Export to Hugging Face Compatible Format
    export_to_huggingface_format(trained_bundle, output_dir="./retro_model")

    summary = {
        "hardware": "Quadro P1000 (FP32)",
        "dataset_routes": n_tot,
        "train_samples": len(idx_train),
        "test_samples": len(idx_test),
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "runtime_seconds": trained_bundle["total_time"],
        "retrieval": retrieval,
        "true_vs_pred_metrics": eval_metrics
    }
    with open("training_results.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
