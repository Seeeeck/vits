import argparse, os
from hubert_model import hubert_soft
from tqdm import tqdm
import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio_dir", required=True, type=Path)
    parser.add_argument("--unit_dir", required=True, type=Path)
    parser.add_argument("--pt_dir", default="./checkpoint/hubert-soft.pt", type=Path)
    parser.add_argument("--extension", default=".wav", type=str)
    parser.add_argument("--sr", default=22050, type=int)
    parser.add_argument("--filelists_path", default="./filelists/train.txt", type=Path)
    args = parser.parse_args()

    print(f"Loading hubert checkpoint")
    hubert = hubert_soft(args.pt_dir)
    print(f"Encoding dataset at {args.audio_dir}")

    lines = []
    for in_path in tqdm(list(args.audio_dir.rglob(f"*{args.extension}"))):
        wav, sr = torchaudio.load(in_path)
        if sr != args.sr:
            wav = resample(wav, sr, args.sr)
        wav = wav.unsqueeze(0).cuda()

        with torch.inference_mode():
            units = hubert.units(wav)

        out_path = args.unit_dir / in_path.relative_to(args.audio_dir)
        lines.append(str(in_path) + "|" + str(out_path.with_suffix(".npy")) + "\n")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())
    with open(args.filelists_path, "w") as f:
        f.writelines(lines)
