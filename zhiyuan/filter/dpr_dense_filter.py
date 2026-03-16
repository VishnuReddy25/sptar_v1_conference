"""Filter weak query-document pairs with a trained DPR model.

This replaces lexical BM25 filtering with dense semantic retrieval.
A weak pair is kept when the target weak document appears in the DPR top-k list
for the generated weak query. The script also exports a soft rank-based quality
score that can be reused for iterative retriever-guided filtering.

Usage example:
python zhiyuan/filter/dpr_dense_filter.py \
  --dataset_name fiqa \
  --train_num 50 \
  --exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2 \
  --output_exp_name llama_7b_100k_fixed_v3_best_llama_prompt_2_dpr_filtered_70 \
  --model_path zhiyuan/retriever/dpr/train/output/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/bert-large-uncased-v1-fiqa \
  --topk 70
"""

import argparse
import csv
import json
import logging
import math
import os
import pathlib
import sys
from os.path import join

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from tqdm import tqdm

cwd = os.getcwd()
if join(cwd, "zhiyuan") not in sys.path:
    sys.path.append(join(cwd, "zhiyuan"))

from data_process import read_weak_json


from typing import Optional


def compute_soft_score(rank: Optional[int]) -> float:
    if rank is None:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="msmarco", type=str)
    parser.add_argument("--train_num", default=50, type=int)
    parser.add_argument("--weak_num", default="100k", type=str)
    parser.add_argument("--exp_name", default="llama_7b_none_5000", type=str)
    parser.add_argument(
        "--output_exp_name",
        default=None,
        type=str,
        help="Output weak-data experiment name used for saved weak query/qrel files",
    )
    parser.add_argument("--topk", default=30, type=int)
    parser.add_argument("--min_soft_score", default=0.0, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--corpus_chunk_size", default=100000, type=int)
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--round_name", default=None, type=str,
                        help="Optional suffix for iterative filtering output folder names")
    args = parser.parse_args()

    xuyang_dir = join(cwd, "xuyang", "data")
    beir_dir = join(cwd, "zhiyuan", "datasets", "raw", "beir")

    input_exp_name = args.exp_name
    output_exp_name = args.output_exp_name or f"{input_exp_name}_dpr_filtered_{args.topk}"

    gen_file_dir = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num)
    weak_q_path = join(gen_file_dir, f"weak_queries_{args.train_num}_{input_exp_name}.jsonl")
    weak_qrel_path = join(gen_file_dir, f"weak_train_{args.train_num}_{input_exp_name}.tsv")
    corpus_path = join(beir_dir, args.dataset_name, "corpus.jsonl")

    run_name = args.round_name or output_exp_name
    log_path = join(pathlib.Path(__file__).parent.absolute(), "output", output_exp_name, str(args.train_num), args.dataset_name, args.weak_num, run_name)
    os.makedirs(log_path, exist_ok=True)

    handler = logging.FileHandler(join(log_path, "test_log.txt"))
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[handler],
    )

    logging.info(f"Filtering weak pairs from input exp: {input_exp_name}")
    logging.info(f"Writing filtered weak pairs to output exp: {output_exp_name}")
    logging.info(f"Using DPR model path: {args.model_path}")
    logging.info("Loading corpus, weak queries, and weak qrels")
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=corpus_path,
        query_file=weak_q_path,
        qrels_file=weak_qrel_path,
    ).load_custom()
    raw_weak_queries = read_weak_json(weak_q_path)

    logging.info(f"Loaded {len(corpus)} corpus docs and {len(queries)} weak pairs")

    model = DRES(
        models.SentenceBERT(args.model_path),
        batch_size=args.batch_size,
        corpus_chunk_size=args.corpus_chunk_size,
    )
    retriever = EvaluateRetrieval(model, k_values=[1, 3, 5, 10, args.topk], score_function="cos_sim")

    logging.info("Running DPR retrieval over weak queries")
    results = retriever.retrieve(corpus, queries)

    filtered_weak_q_path = join(gen_file_dir, f"weak_queries_{args.train_num}_{output_exp_name}.jsonl")
    filtered_weak_qrel_path = join(gen_file_dir, f"weak_train_{args.train_num}_{output_exp_name}.tsv")
    soft_score_path = join(gen_file_dir, f"weak_train_{args.train_num}_{output_exp_name}_scores.tsv")

    for out_path in [filtered_weak_q_path, filtered_weak_qrel_path, soft_score_path]:
        if os.path.exists(out_path):
            os.remove(out_path)
            logging.info(f"Deleted old output: {out_path}")

    kept = {}
    kept_qrels = {}

    with open(soft_score_path, "w", newline="") as score_file:
        writer = csv.writer(score_file, delimiter="\t")
        writer.writerow(["query-id", "target-doc-id", "rank", "soft_score", "kept"])

        for qid, doc_scores in tqdm(results.items()):
            ranked_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            ranked_doc_ids = [doc_id for doc_id, _ in ranked_doc_ids[: args.topk]]
            weak_doc_id = list(qrels[qid].keys())[0]

            rank = None
            if weak_doc_id in ranked_doc_ids:
                rank = ranked_doc_ids.index(weak_doc_id) + 1

            soft_score = compute_soft_score(rank)
            keep = rank is not None and soft_score >= args.min_soft_score
            writer.writerow([qid, weak_doc_id, rank if rank is not None else -1, f"{soft_score:.8f}", int(keep)])

            if keep:
                kept[qid] = raw_weak_queries[qid]
                kept_qrels[qid] = qrels[qid]

    with open(filtered_weak_qrel_path, "w") as qrel_file:
        qrel_file.write("query-id\tcorpus-id\tscore\n")
        for qid, val in kept_qrels.items():
            for doc_id in val:
                qrel_file.write(f"{qid}\t{doc_id}\t1\n")

    with open(filtered_weak_q_path, "w") as query_file:
        for weak_query in kept.values():
            json.dump(weak_query, query_file)
            query_file.write("\n")

    keep_ratio = len(kept) / max(1, len(queries))
    logging.info(f"After DPR filter: {len(kept)} / {len(queries)} kept ({keep_ratio:.2%})")
    logging.info(f"Filtered weak queries: {filtered_weak_q_path}")
    logging.info(f"Filtered weak qrels: {filtered_weak_qrel_path}")
    logging.info(f"Soft scores: {soft_score_path}")


if __name__ == "__main__":
    main()
