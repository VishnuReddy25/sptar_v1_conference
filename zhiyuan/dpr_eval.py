'''
import subprocess
import argparse


def multirun(args):
    # Convert args namespace to dictionary
    arg_dict = vars(args)

    for exp in arg_dict["exp_names"]:
        print(f"GPU {arg_dict['gpu_id']} Training: {arg_dict['dataset_name']} on {exp}")

        # Uncomment below if training is needed
        if arg_dict["version"] == "v1":
            train_command = [
                "python", "zhiyuan/retriever/dpr/train/train_sbert.py",
                "--dataset_name", arg_dict["dataset_name"],
                "--train_num", str(arg_dict["train_num"]),
                "--weak_num", arg_dict["weak_num"],
                "--exp_name", exp
            ]
        elif arg_dict["version"] == "v2":
            train_command = [
                "python", "zhiyuan/retriever/dpr/train/train_sbert_BM25_hardnegs.py",
                "--dataset_name", arg_dict["dataset_name"],
                "--train_num", str(arg_dict["train_num"]),
                "--weak_num", arg_dict["weak_num"],
                "--exp_name", exp
            ]
        print("Running training command:", " ".join(train_command))
        subprocess.call(train_command)

        # Always run evaluation
        eval_command = [
            "python", "zhiyuan/retriever/dpr/eval/evaluate_sbert.py",
            "--dataset_name", arg_dict["dataset_name"],
            "--train_num", str(arg_dict["train_num"]),
            "--exp_name", exp,
            "--dpr_v", arg_dict["version"]
        ]
        print("Running evaluation command:", " ".join(eval_command), "\n")
        subprocess.call(eval_command)


def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation scripts.')
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--version", type=str, choices=["v1", "v2"], required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--train_num", type=int, required=True)
    parser.add_argument("--weak_num", type=str, required=True)
    parser.add_argument("--exp_names", nargs='+', required=True, help="List of experiment names")
    
    args = parser.parse_args()
    multirun(args)


if __name__ == "__main__":
    main()
'''

import subprocess
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Run DPR training and evaluation.')

    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--version", type=str, choices=["v1", "v2"], required=True)
    parser.add_argument("--gpu_id", type=int, required=True)
    parser.add_argument("--train_num", type=int, required=True)
    parser.add_argument("--weak_num", type=str, required=True)
    parser.add_argument("--exps", type=str, required=True)

    # ============================
    # NEW ARGUMENTS
    # ============================
    parser.add_argument(
        '--iter_round',
        type=int,
        default=0,
        help='Iteration round. 0=original BM25 filter, 1+=retriever filter'
    )

    parser.add_argument(
        '--use_retriever_filter',
        action='store_true',
        help='Use trained retriever to filter weak data instead of BM25'
    )

    args = parser.parse_args()

    # ===============================================
    # ITERATIVE RETRIEVER FILTERING (NEW BLOCK)
    # ===============================================
    if args.use_retriever_filter and args.iter_round > 0:

        from zhiyuan.filter.retriever_filter import run_retriever_filter

        # Path to retriever trained in previous round
        prev_model_path = (
            f"zhiyuan/retriever/dpr/train/output/"
            f"{args.exps}_round{args.iter_round-1}/"
        )

        # Raw weak data (never changes)
        raw_weak_data = (
            f"zhiyuan/datasets/weak/"
            f"{args.dataset_name}/raw_100k.jsonl"
        )

        # Corpus path
        corpus_path = (
            f"zhiyuan/datasets/raw/beir/"
            f"{args.dataset_name}/corpus_filtered.jsonl"
        )

        # Output filtered file for this round
        filtered_output = (
            f"zhiyuan/datasets/weak/"
            f"{args.dataset_name}/"
            f"retriever_filtered_round{args.iter_round}.jsonl"
        )

        print("=" * 60)
        print(f"Round {args.iter_round}: Filtering weak data "
              f"using retriever from round {args.iter_round-1}...")
        print("=" * 60)

        run_retriever_filter(
            model_path=prev_model_path,
            weak_data_path=raw_weak_data,
            corpus_path=corpus_path,
            output_path=filtered_output,
            top_k=50
        )

        # Update experiment name for this round
        args.exps = f"{args.exps}_round{args.iter_round}"

    # ===============================================
    # TRAIN COMMAND
    # ===============================================
    if args.version == "v1":
        train_script = "zhiyuan/retriever/dpr/train/train_sbert.py"
    else:
        train_script = "zhiyuan/retriever/dpr/train/train_sbert_BM25_hardnegs.py"

    train_command = [
        "python", train_script,
        "--dataset_name", args.dataset_name,
        "--train_num", str(args.train_num),
        "--weak_num", args.weak_num,
        "--exp_name", args.exps
    ]

    print("\nRunning training command:")
    print(" ".join(train_command))
    subprocess.call(train_command)

    # ===============================================
    # EVALUATION COMMAND
    # ===============================================
    eval_command = [
        "python",
        "zhiyuan/retriever/dpr/eval/evaluate_sbert.py",
        "--dataset_name", args.dataset_name,
        "--train_num", str(args.train_num),
        "--exp_name", args.exps,
        "--dpr_v", args.version
    ]

    print("\nRunning evaluation command:")
    print(" ".join(eval_command))
    subprocess.call(eval_command)


if __name__ == "__main__":
    main()
