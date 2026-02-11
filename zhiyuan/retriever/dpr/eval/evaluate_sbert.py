import os
import sys
import json
import argparse
import logging
import pathlib
from os.path import join
import random

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval

# Set up local import paths
cwd = os.getcwd()
zhiyuan_path = join(cwd, "zhiyuan")
xuyang_path = join(cwd, "xuyang")

if zhiyuan_path not in sys.path:
    sys.path.append(zhiyuan_path)
if xuyang_path not in sys.path:
    sys.path.append(xuyang_path)

# Local imports
from data_process import load_dl, merge_queries, extract_results

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=False, default="msmarco")
parser.add_argument('--train_num', type=int, required=False, default=50)
parser.add_argument('--dpr_v', type=str, choices=["v1", "v2"], default="v1")
parser.add_argument('--exp_name', type=str, default="no_aug")
args = parser.parse_args()

# Paths
# model_name = "bert-base-uncased"
model_name = "bert-large-uncased"
model_save_path = os.path.join(
    pathlib.Path(__file__).parent.parent.absolute(),
    "train", "output", args.exp_name, str(args.train_num),
    f"{model_name}-{args.dpr_v}-{args.dataset_name}"
)
os.makedirs(model_save_path, exist_ok=True)
model_save_path = r"/home/aiml_user/vishnu/sptar_v1/zhiyuan/retriever/dpr/train/output/llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70/50/bert-large-uncased-v1-fiqa"
# Setup logging
log_file = join(model_save_path, "test_log.txt")
handler = logging.FileHandler(log_file)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[handler]
)

# BEIR paths
data_dir = join(zhiyuan_path, "datasets")
raw_dir = join(data_dir, "raw")
beir_dir = join(raw_dir, "beir")

# Load model
model = DRES(models.SentenceBERT(model_save_path), batch_size=256, corpus_chunk_size=100000)
retriever = EvaluateRetrieval(model, k_values=[1, 3, 5, 10, 100, 300, 500, 1000], score_function="cos_sim")
print(model)
# Load corpus and queries
if args.dataset_name == "msmarco":
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="dev")
    queries_19, qrels_19, qrels_binary_19 = load_dl(join(beir_dir, "TREC_DL_2019"))
    queries_20, qrels_20, qrels_binary_20 = load_dl(join(beir_dir, "TREC_DL_2020"))
else:
    corpus, queries, qrels = GenericDataLoader(join(beir_dir, args.dataset_name)).load(split="test")

# Sample for debug (optional)
# corpus = dict(random.sample(corpus.items(), 100000))

# Prepare evaluation sets
tobe_eval = {}

if args.dataset_name == "msmarco":
    ms_queries = merge_queries(queries, queries_19, queries_20)
    ms_results = retriever.retrieve(corpus, ms_queries)
    results, results_19, results_20 = extract_results(ms_results)
    tobe_eval["dl2019"] = (qrels_19, results_19, qrels_binary_19)
    tobe_eval["dl2020"] = (qrels_20, results_20, qrels_binary_20)
else:
    results = retriever.retrieve(corpus, queries)

tobe_eval[args.dataset_name] = (qrels, results, "pad")

# Run evaluation
for dataset_name in tobe_eval:
    qrels, results, qrels_binary = tobe_eval[dataset_name]
    logging.info(f"Retriever evaluation for dataset {dataset_name}")

    # Evaluate returns 4 values
    ndcg, map, recall, score_per_query = retriever.evaluate(qrels, results, retriever.k_values)

    # Special handling for TREC-DL binary relevance
    if dataset_name in ["dl2019", "dl2020"]:
        _, map_bin, recall_bin, score_per_query_override = retriever.evaluate(qrels_binary, results, retriever.k_values)
        for key in score_per_query.keys():
            if "MAP" in key:
                score_per_query[key] = score_per_query_override.get(key, 0.0)
            if "Recall" in key:
                score_per_query[key] = score_per_query_override.get(key, 0.0)
    else:
        # MRR is returned as a single dict
        mrr_score = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
        for key in mrr_score:
            score_per_query[key] = mrr_score[key]

    # Log evaluation results
    for metric_dict in [ndcg, map, recall]:
        logging.info("\n")
        for k in metric_dict:
            logging.info(f"{k}: {metric_dict[k]:.4f}")

    # Save to file
    output_file = join(model_save_path, f"{dataset_name}.json")
    with open(output_file, "w") as f:
        json.dump(score_per_query, f)
