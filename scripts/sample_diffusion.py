# scripts/sample_diffusion.py
import argparse
import torch
import os
from pathlib import Path
from src.models.diffusion.sampler import sample_and_decode

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--preprocessor_dir", type=str, required=True,
                   help="path to preprocessor dir with metadata.json and joblib files (e.g., data/processed/adult/preprocessor)")
    p.add_argument("--out_csv", type=str, default="results/samples/generated.csv")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--method", type=str, default="ddim", choices=["ddim", "ddpm"])
    p.add_argument("--ddim_steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--embed_dim", type=int, default=8)
    p.add_argument("--hidden_dim", type=int, default=128) # Added argument
    p.add_argument("--num_layers", type=int, default=4) # Added argument
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    sample_and_decode(checkpoint_path=args.checkpoint,
                      preprocessor_dir=args.preprocessor_dir,
                      output_csv=args.out_csv,
                      num_samples=args.num_samples,
                      method=args.method,
                      ddim_steps=args.ddim_steps,
                      eta=args.eta,
                      device=torch.device(args.device),
                      embed_dim=args.embed_dim,
                      hidden_dim=args.hidden_dim, # Pass argument
                      num_layers=args.num_layers) # Pass argument
