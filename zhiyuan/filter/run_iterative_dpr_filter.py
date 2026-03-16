"""Run iterative DPR weak-data filtering + retraining rounds.

This script performs the full loop requested by the SPTAR iterative workflow:
1) Filter weak pairs using DPR
2) Retrain DPR on filtered weak pairs
3) Use newly trained DPR as next round's filter model

Example:
python zhiyuan/filter/run_iterative_dpr_filter.py \
  --dataset_name fiqa \
  --train_num 50 \
  --weak_num 100k \
  --raw_exp_name llama_7b_100k_fixed_v4_best_llama_prompt_3 \
  --init_model_path zhiyuan/retriever/dpr/train/output/no_aug/50/bert-large-uncased-v1-fiqa \
  --topk 30 \
  --rounds 2
"""

import argparse
import subprocess
import sys
from pathlib import Path


from typing import List


def run_cmd(cmd: List[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def model_output_path(repo_root: Path, exp_name: str, train_num: int, dataset_name: str) -> Path:
    return repo_root / "zhiyuan" / "retriever" / "dpr" / "train" / "output" / exp_name / str(train_num) / f"bert-large-uncased-v1-{dataset_name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="fiqa", type=str)
    parser.add_argument("--train_num", default=50, type=int)
    parser.add_argument("--weak_num", default="100k", type=str)
    parser.add_argument("--raw_exp_name", required=True, type=str, help="Raw weak-data exp name to filter each round")
    parser.add_argument("--init_model_path", required=True, type=str, help="DPR checkpoint used in round 1 filtering")
    parser.add_argument("--topk", default=30, type=int)
    parser.add_argument("--rounds", default=2, type=int)
    parser.add_argument("--num_epochs", default=2, type=int)
    parser.add_argument("--product", default="cosine", choices=["cosine", "dot"])
    parser.add_argument("--min_soft_score", default=0.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--corpus_chunk_size", default=100000, type=int)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    python_bin = sys.executable

    current_filter_model = Path(args.init_model_path)

    for round_id in range(1, args.rounds + 1):
        round_exp_name = f"{args.raw_exp_name}_dpr_r{round_id}_top{args.topk}"
        round_name = f"round_{round_id}"

        print(f"\n===== Iteration Round {round_id}/{args.rounds} =====", flush=True)
        print(f"Filter model: {current_filter_model}", flush=True)
        print(f"Filtered weak exp: {round_exp_name}", flush=True)

        filter_cmd = [
            python_bin,
            str(repo_root / "zhiyuan" / "filter" / "dpr_dense_filter.py"),
            "--dataset_name", args.dataset_name,
            "--train_num", str(args.train_num),
            "--weak_num", args.weak_num,
            "--exp_name", args.raw_exp_name,
            "--output_exp_name", round_exp_name,
            "--model_path", str(current_filter_model),
            "--topk", str(args.topk),
            "--min_soft_score", str(args.min_soft_score),
            "--batch_size", str(args.batch_size),
            "--corpus_chunk_size", str(args.corpus_chunk_size),
            "--round_name", round_name,
        ]
        run_cmd(filter_cmd)

        train_cmd = [
            python_bin,
            "-m", "zhiyuan.retriever.dpr.train.train_sbert",
            "--dataset_name", args.dataset_name,
            "--train_num", str(args.train_num),
            "--weak_num", args.weak_num,
            "--num_epochs", str(args.num_epochs),
            "--product", args.product,
            "--exp_name", round_exp_name,
        ]
        run_cmd(train_cmd)

        current_filter_model = model_output_path(repo_root, round_exp_name, args.train_num, args.dataset_name)

    print("\nAll iterative rounds completed.", flush=True)


if __name__ == "__main__":
    main()
