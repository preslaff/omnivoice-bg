# Copyright    2026  OmniVoice-BG
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Train the Accentor on the text-only marked corpus (mark_chitanka_text.py).

Loss = BCE on labeled vowel positions only: stress-marked words supervise
their vowels 0/1, zero-label function words supervise all-zero, and vowels
inside `mask_words` spans (ambiguous/OOV) are EXCLUDED — their labels arrive
later from human speech (parliament + real audiobooks); this run is the
text-only PRETRAIN of that curriculum.

Metrics (val split, deterministic 1% by id):
  val_loss   masked BCE
  ser        word-level stress error rate on fully-labeled single-stress words
             (predicted argmax vowel != target) — the paper's headline metric
  zero_err   fully-labeled zero-stress (function) words with any vowel >= 0.5

Usage:
    ./venv/bin/python train_accentor.py \
        --data data/chitanka/marked.jsonl --out exp/accentor_text \
        --steps 100000 --batch-size 256
"""
import argparse
import json
import math
import os
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from accentor_model import ACUTE, VOCAB, VOWELS, Accentor, encode_text, n_params

LETTER_ID_MAX = 31  # ids 2..31 are the 30 BG letters (see accentor_model.VOCAB)


def row_arrays(row):
    """marked.jsonl row -> (ids, caps, target, lossmask) uint8 arrays."""
    text = row["text"]
    ids, caps = encode_text(text)
    n = len(text)
    tgt = np.zeros(n, dtype=np.uint8)
    ti = 0
    for ch in row["marked"]:
        if ch == ACUTE:
            tgt[ti - 1] = 1
        else:
            ti += 1
    if ti != n:  # marked/text desync — skip row
        return None
    low = text.lower().replace("ѝ", "и")
    lmask = np.frombuffer(
        bytes(1 if c in VOWELS else 0 for c in low), dtype=np.uint8).copy()
    for s, e in row["mask_words"]:
        lmask[s:e] = 0
    return (np.asarray(ids, dtype=np.uint8), np.asarray(caps, dtype=np.uint8),
            tgt, lmask)


def load_corpus(path, max_rows=0, val_pct=1):
    """-> dict(split -> (flat ids/caps/tgt/lmask, offsets int64))."""
    buf = {s: dict(ids=[], caps=[], tgt=[], lmask=[], off=[0]) for s in
           ("train", "val")}
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc="load", unit=" rows"):
            row = json.loads(line)
            arrs = row_arrays(row)
            if arrs is None:
                continue
            split = "val" if zlib.crc32(row["id"].encode()) % 100 < val_pct \
                else "train"
            b = buf[split]
            for k, a in zip(("ids", "caps", "tgt", "lmask"), arrs):
                b[k].append(a)
            b["off"].append(b["off"][-1] + len(arrs[0]))
            n += 1
            if max_rows and n >= max_rows:
                break
    out = {}
    for split, b in buf.items():
        out[split] = {k: np.concatenate(b[k]) if b[k] else np.zeros(0, np.uint8)
                      for k in ("ids", "caps", "tgt", "lmask")}
        out[split]["off"] = np.asarray(b["off"], dtype=np.int64)
    return out


def batches(data, order, batch_size, device):
    off = data["off"]
    for i in range(0, len(order), batch_size):
        idx = order[i : i + batch_size]
        rows = [(off[j], off[j + 1]) for j in idx]
        T = max(e - s for s, e in rows)
        B = len(rows)
        pad = {k: np.zeros((B, T), dtype=np.uint8)
               for k in ("ids", "caps", "tgt", "lmask")}
        attn = np.zeros((B, T), dtype=bool)
        for bi, (s, e) in enumerate(rows):
            L = e - s
            attn[bi, :L] = True
            for k in pad:
                pad[k][bi, :L] = data[k][s:e]
        yield (torch.from_numpy(pad["ids"]).long().to(device),
               torch.from_numpy(pad["caps"]).long().to(device),
               torch.from_numpy(attn).to(device),
               torch.from_numpy(pad["tgt"]).float().to(device),
               torch.from_numpy(pad["lmask"]).bool().to(device))


def masked_bce(logits, tgt, lmask):
    per = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    return (per * lmask).sum() / lmask.sum().clamp(min=1)


@torch.no_grad()
def evaluate(model, data, batch_size, device, max_rows=20000):
    model.eval()
    n_rows = len(data["off"]) - 1
    order = np.arange(min(n_rows, max_rows))
    tot_loss = tot_n = 0.0
    ser_err = ser_n = zero_err = zero_n = 0
    for ids, caps, attn, tgt, lmask in batches(data, order, batch_size, device):
        logits = model(ids, caps, attn)
        tot_loss += masked_bce(logits, tgt, lmask).item() * lmask.sum().item()
        tot_n += lmask.sum().item()
        probs = torch.sigmoid(logits).cpu().numpy()
        ids_np = ids.cpu().numpy()
        tgt_np, lm_np = tgt.cpu().numpy(), lmask.cpu().numpy()
        attn_np = attn.cpu().numpy()
        for b in range(ids_np.shape[0]):
            L = attn_np[b].sum()
            is_letter = (ids_np[b, :L] >= 2) & (ids_np[b, :L] <= LETTER_ID_MAX)
            s = 0
            while s < L:
                if not is_letter[s]:
                    s += 1
                    continue
                e = s
                while e < L and is_letter[e]:
                    e += 1
                vpos = [i for i in range(s, e) if lm_np[b, i]]
                # fully-labeled words only (mask_words zeroes lmask wholesale)
                n_vow_all = sum(
                    1 for i in range(s, e)
                    if VOCAB[ids_np[b, i]] in VOWELS)
                if vpos and len(vpos) == n_vow_all:
                    n_str = int(tgt_np[b, vpos].sum())
                    if n_str == 1:
                        ser_n += 1
                        pred = max(vpos, key=lambda i: probs[b, i])
                        if tgt_np[b, pred] != 1:
                            ser_err += 1
                    elif n_str == 0:
                        zero_n += 1
                        if any(probs[b, i] >= 0.5 for i in vpos):
                            zero_err += 1
                s = e
    model.train()
    return (tot_loss / max(tot_n, 1), ser_err / max(ser_n, 1), ser_n,
            zero_err / max(zero_n, 1), zero_n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/chitanka/marked.jsonl")
    ap.add_argument("--out", default="exp/accentor_text")
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--val-rows", type=int, default=20000)
    ap.add_argument("--max-rows", type=int, default=0, help="pilot cap")
    ap.add_argument("--separable", action="store_true",
                    help="paper-exact depthwise-separable convs (~6.7M params)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to CONTINUE (restores optimizer + step)")
    ap.add_argument("--init", default=None,
                    help="checkpoint to initialize WEIGHTS from (fine-tune): "
                         "fresh optimizer + LR schedule, step starts at 0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    data = load_corpus(args.data, args.max_rows)
    n_train = len(data["train"]["off"]) - 1
    n_val = len(data["val"]["off"]) - 1
    print(f"rows: train {n_train:,} / val {n_val:,}")

    if args.init:
        ick = torch.load(args.init, map_location=device)
        cfg = dict(ick["config"])
        cfg["kernel"] = tuple(cfg["kernel"])
        model = Accentor(**cfg).to(device)
        model.load_state_dict(ick["model"])
        print(f"initialized weights from {args.init} (pretrain step "
              f"{ick.get('step')}, SER {ick.get('best_ser', float('nan')):.3%})")
    else:
        model = Accentor(separable=args.separable).to(device)
    print(f"params: {n_params(model)/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    step = 0
    best_ser = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best_ser = ck["step"], ck.get("best_ser", best_ser)
        print(f"resumed from {args.resume} @ step {step}")

    metrics_path = os.path.join(args.out, "metrics.jsonl")
    mf = open(metrics_path, "a", encoding="utf-8")
    tb = None
    try:  # repo ships the tensorboardX writer; viewer installed separately
        from tensorboardX import SummaryWriter
        tb = SummaryWriter(os.path.join(args.out, "tensorboard"))
    except ImportError:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(os.path.join(args.out, "tensorboard"))
        except ImportError:
            print("tensorboard(X) not available — metrics.jsonl only")
    order = np.random.permutation(n_train)
    it = iter(batches(data["train"], order, args.batch_size, device))
    pbar = tqdm(total=args.steps, initial=step, desc="train")
    t0 = time.time()
    while step < args.steps:
        try:
            ids, caps, attn, tgt, lmask = next(it)
        except StopIteration:
            order = np.random.permutation(n_train)
            it = iter(batches(data["train"], order, args.batch_size, device))
            continue
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss = masked_bce(model(ids, caps, attn), tgt, lmask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        pbar.update(1)
        if step % 50 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             lr=f"{lr_at(step):.1e}")
            if tb:
                tb.add_scalar("train/loss", loss.item(), step)
                tb.add_scalar("train/learning_rate", lr_at(step), step)
        if step % args.eval_every == 0 or step == args.steps:
            vl, ser, sn, ze, zn = evaluate(model, data["val"],
                                           args.batch_size, device,
                                           args.val_rows)
            rec = dict(step=step, train_loss=round(loss.item(), 5),
                       val_loss=round(vl, 5), ser=round(ser, 5), ser_n=sn,
                       zero_err=round(ze, 5), zero_n=zn,
                       wall=round(time.time() - t0, 1))
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            if tb:
                tb.add_scalar("eval/loss", vl, step)
                tb.add_scalar("eval/ser", ser, step)
                tb.add_scalar("eval/zero_err", ze, step)
            tqdm.write(f"step {step}: val_loss {vl:.4f}  SER {ser:.3%} "
                       f"(n={sn:,})  zero_err {ze:.3%} (n={zn:,})")
            ck = dict(model=model.state_dict(), opt=opt.state_dict(),
                      step=step, best_ser=best_ser, config=model.config)
            torch.save(ck, os.path.join(args.out, "last.pt"))
            if ser < best_ser:
                best_ser = ser
                ck["best_ser"] = ser
                torch.save(ck, os.path.join(args.out, "best.pt"))
    pbar.close()
    if tb:
        tb.close()
    print(f"done. best SER {best_ser:.3%}  checkpoints in {args.out}/")


if __name__ == "__main__":
    main()
