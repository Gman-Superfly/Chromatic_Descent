import argparse
import json
import os
from pathlib import Path

from chromatic.train import train_many
from chromatic.embed import build_embeddings
from chromatic.ensemble import evaluate_ensembles


def parse_args():
    parser = argparse.ArgumentParser(description="Chromatic Descent experiment")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10"], help="Dataset name")
    parser.add_argument("--data_root", default="./data", help="Dataset root")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per run")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seeds", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    parser.add_argument("--fast", action="store_true", help="Use a tiny subset for quick smoke test")
    parser.add_argument("--repel", action="store_true", help="Enable repelled descent")
    parser.add_argument("--repel_lambda", type=float, default=0.1, help="Strength of repulsion term")
    parser.add_argument("--repel_warmup", type=int, default=2, help="Epochs before enabling repulsion")
    parser.add_argument("--out_dir", default="./artifacts", help="Where to store artifacts")
    return parser.parse_args()


def main():
    args = parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)

    run_infos = train_many(
        dataset=args.dataset,
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        seeds=args.seeds,
        device=args.device,
        fast=args.fast,
        repel=args.repel,
        repel_lambda=args.repel_lambda,
        repel_warmup=args.repel_warmup,
        runs_dir=runs_dir,
        out_dir=Path(args.out_dir),
    )

    emb_info = build_embeddings(run_infos, out_dir=Path(args.out_dir))
    report = evaluate_ensembles(run_infos, out_dir=Path(args.out_dir))

    summary = {"runs": run_infos, "embeddings": emb_info, "ensemble": report}
    with open(Path(args.out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved summary to", Path(args.out_dir) / "summary.json")


if __name__ == "__main__":
    main()


