"""Single-GPU (no DDP) Kronos fine-tune wrapper.

Replicates Kronos's `finetune/train_tokenizer.py` + `train_predictor.py` training
loops without DDP (DDP on Windows + torchrun is flaky). Loads same config,
same QlibDataset, same model. Just removes distributed wrappers.

Usage:
  python kronos_train_single_gpu.py --stage tokenizer
  python kronos_train_single_gpu.py --stage predictor
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


KRONOS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "kronos" / "repo"
sys.path.insert(0, str(KRONOS_DIR / "finetune"))
sys.path.insert(0, str(KRONOS_DIR))


def train_tokenizer(config, device):
    from dataset_numpy import QlibDataset
    from model.kronos import KronosTokenizer

    print("[tokenizer] loading datasets ...")
    train_ds = QlibDataset('train')
    val_ds = QlibDataset('val')
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)
    print(f"  train_loader steps/epoch: {len(train_loader)}, val steps: {len(val_loader)}")

    print(f"[tokenizer] loading pretrained: {config.pretrained_tokenizer_path}")
    model = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config.tokenizer_learning_rate,
                                   weight_decay=config.adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=config.tokenizer_learning_rate,
        steps_per_epoch=len(train_loader), epochs=config.epochs,
        pct_start=0.03, div_factor=10,
    )

    save_dir = Path(config.save_path) / config.tokenizer_save_folder_name
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    t_total = time.time()

    for epoch in range(config.epochs):
        t_epoch = time.time()
        model.train()
        train_ds.set_epoch_seed(epoch)
        running_loss = 0.0
        for i, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            zs, bsq_loss, _, _ = model(batch_x)
            z_pre, z = zs
            recon_loss_pre = F.mse_loss(z_pre, batch_x)
            recon_loss_all = F.mse_loss(z, batch_x)
            recon_loss = recon_loss_pre + recon_loss_all
            loss = (recon_loss + bsq_loss) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if (i + 1) % config.log_interval == 0:
                print(f"  ep{epoch+1}/{config.epochs} step{i+1}/{len(train_loader)} "
                      f"lr={optimizer.param_groups[0]['lr']:.6f} loss={loss.item():.4f}",
                      flush=True)

        # Validation
        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                zs, _, _, _ = model(batch_x)
                _, z = zs
                vl = F.mse_loss(z, batch_x).item()
                val_loss_sum += vl * batch_x.size(0)
                val_count += batch_x.size(0)
        avg_val = val_loss_sum / max(val_count, 1)
        print(f"--- epoch {epoch+1} | val_loss={avg_val:.4f} | "
              f"epoch_time={time.time()-t_epoch:.0f}s | total={time.time()-t_total:.0f}s ---",
              flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = save_dir / "checkpoints" / "best_model"
            model.save_pretrained(str(save_path))
            print(f"  saved best to {save_path} (val_loss={avg_val:.4f})", flush=True)

    summary = {"best_val_loss": best_val_loss, "total_time_s": time.time() - t_total,
               "epochs": config.epochs}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[tokenizer] DONE. best_val={best_val_loss:.4f}, time={time.time()-t_total:.0f}s")


def train_predictor(config, device):
    from dataset_numpy import QlibDataset
    from model.kronos import Kronos, KronosTokenizer

    print("[predictor] loading datasets ...")
    train_ds = QlibDataset('train')
    val_ds = QlibDataset('val')
    print(f"  train={len(train_ds):,}  val={len(val_ds):,}")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)

    # Load fine-tuned tokenizer (or pretrained if not done)
    tok_path = Path(config.finetuned_tokenizer_path)
    if tok_path.exists():
        print(f"[predictor] loading fine-tuned tokenizer: {tok_path}")
        tokenizer = KronosTokenizer.from_pretrained(str(tok_path))
    else:
        print(f"[predictor] using pretrained tokenizer: {config.pretrained_tokenizer_path}")
        tokenizer = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    tokenizer.to(device).eval()
    for p in tokenizer.parameters():
        p.requires_grad = False

    print(f"[predictor] loading pretrained predictor: {config.pretrained_predictor_path}")
    model = Kronos.from_pretrained(config.pretrained_predictor_path)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=config.predictor_learning_rate,
                                   weight_decay=config.adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, max_lr=config.predictor_learning_rate,
        steps_per_epoch=len(train_loader), epochs=config.epochs,
        pct_start=0.03, div_factor=10,
    )

    save_dir = Path(config.save_path) / config.predictor_save_folder_name
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    t_total = time.time()

    for epoch in range(config.epochs):
        t_epoch = time.time()
        model.train()
        train_ds.set_epoch_seed(epoch)
        for i, (batch_x, batch_t) in enumerate(train_loader):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_t = batch_t.to(device, non_blocking=True)

            with torch.no_grad():
                token_in_pre, token_in_post = tokenizer.encode(batch_x, half=True)
            logits = model(token_in_pre, token_in_post, batch_t)
            # logits is tuple (s1_logits, s2_logits) — standard cross-entropy
            s1_logits, s2_logits = logits
            # Targets are shifted tokens (next-token prediction)
            s1_tgt = token_in_pre[:, 1:].long()
            s2_tgt = token_in_post[:, 1:].long()
            s1_pred = s1_logits[:, :-1]
            s2_pred = s2_logits[:, :-1]

            loss_s1 = F.cross_entropy(s1_pred.reshape(-1, s1_pred.size(-1)), s1_tgt.reshape(-1))
            loss_s2 = F.cross_entropy(s2_pred.reshape(-1, s2_pred.size(-1)), s2_tgt.reshape(-1))
            loss = (loss_s1 + loss_s2) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()
            if (i + 1) % config.log_interval == 0:
                print(f"  ep{epoch+1}/{config.epochs} step{i+1}/{len(train_loader)} "
                      f"lr={optimizer.param_groups[0]['lr']:.6f} loss={loss.item():.4f} "
                      f"(s1={loss_s1.item():.3f} s2={loss_s2.item():.3f})",
                      flush=True)

        # Validation
        model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_t in val_loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_t = batch_t.to(device, non_blocking=True)
                token_in_pre, token_in_post = tokenizer.encode(batch_x, half=True)
                s1_logits, s2_logits = model(token_in_pre, token_in_post, batch_t)
                s1_tgt = token_in_pre[:, 1:].long()
                s2_tgt = token_in_post[:, 1:].long()
                loss_s1 = F.cross_entropy(
                    s1_logits[:, :-1].reshape(-1, s1_logits.size(-1)),
                    s1_tgt.reshape(-1)
                )
                loss_s2 = F.cross_entropy(
                    s2_logits[:, :-1].reshape(-1, s2_logits.size(-1)),
                    s2_tgt.reshape(-1)
                )
                vl = ((loss_s1 + loss_s2) / 2).item()
                val_loss_sum += vl * batch_x.size(0)
                val_count += batch_x.size(0)
        avg_val = val_loss_sum / max(val_count, 1)
        print(f"--- epoch {epoch+1} | val_loss={avg_val:.4f} | "
              f"epoch_time={time.time()-t_epoch:.0f}s | total={time.time()-t_total:.0f}s ---",
              flush=True)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            save_path = save_dir / "checkpoints" / "best_model"
            model.save_pretrained(str(save_path))
            print(f"  saved best to {save_path} (val_loss={avg_val:.4f})", flush=True)

    summary = {"best_val_loss": best_val_loss, "total_time_s": time.time() - t_total,
               "epochs": config.epochs}
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[predictor] DONE. best_val={best_val_loss:.4f}, time={time.time()-t_total:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tokenizer", "predictor"])
    args = ap.parse_args()

    from config import Config
    config = Config()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if args.stage == "tokenizer":
        train_tokenizer(config, device)
    else:
        train_predictor(config, device)


if __name__ == "__main__":
    main()
