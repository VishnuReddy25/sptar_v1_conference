'''
This examples show how to train a basic Bi-Encoder for any BEIR dataset without any mined hard negatives or triplets.

The queries and passages are passed independently to the transformer network to produce fixed sized embeddings.
These embeddings can then be compared using cosine-similarity to find matching passages for a given query.

For training, we use MultipleNegativesRankingLoss. There, we pass pairs in the format:
(query, positive_passage). Other positive passages within a single batch becomes negatives given the pos passage.

We do not mine hard negatives or train triplets in this example.

Running this script:
python train_sbert.py
'''
'''

import torch
from sentence_transformers import losses, models, SentenceTransformer
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
import argparse
from os.path import join, dirname, abspath
import math
import sys
####
print("Started",flush=True)
print("Started without flush")

zhiyuan_path = dirname(dirname(dirname(dirname(abspath(__file__)))))
if zhiyuan_path not in sys.path:
    sys.path.append(zhiyuan_path)

from weak_data_loader import WeakDataLoader

data_dir = join(zhiyuan_path, "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = join(dirname(zhiyuan_path), "xuyang", "data")

#### Download nfcorpus.zip dataset and unzip the dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', required=False, default="msmarco", type=str)
parser.add_argument('--num_epochs', required=False, default=2, type=int)
parser.add_argument('--train_num', required=False, default=50, type=int)
parser.add_argument('--weak_num', required=False, default="5000", type=str)
parser.add_argument('--product', required=False, default="cosine", type=str)
parser.add_argument('--exp_name', required=False, default="no_aug", type=str)
args = parser.parse_args()
#### Provide model save path
# model_name = "bert-base-uncased" 
model_name = "bert-large-uncased" 
model_save_path = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", args.exp_name, str(args.train_num), "{}-v1-{}".format(model_name, args.dataset_name))
os.makedirs(model_save_path, exist_ok=True)
#### Just some code to print debug information to stdout
fh = logging.FileHandler(join(model_save_path, "log.txt"))
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[fh, ch])
#### /print debug information to stdout
#### Provide the data_path where nfcorpus has been downloaded and unzipped
if args.exp_name == "no_aug":
    corpus, queries, qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv")).load_custom()
else:
    # add support for loading weak data and ori train as new train
    weak_query_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_queries_{args.train_num}_{args.exp_name}.jsonl")
    weak_qrels_file = join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", args.weak_num, f"weak_train_{args.train_num}_{args.exp_name}.tsv")
    corpus, queries, qrels = WeakDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(xuyang_dir, f"{args.dataset_name}_{args.train_num}", f"prompt_tuning_{args.train_num}.tsv"), weak_query_file=weak_query_file, weak_qrels_file=weak_qrels_file).load_weak_custom()
#### Please Note not all datasets contain a dev split, comment out the line if such the case
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(corpus_file=join(beir_dir, args.dataset_name, f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"), query_file=join(beir_dir, args.dataset_name, "queries.jsonl"), qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")).load_custom()

#### Provide any sentence-transformers or HF model
word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

#### Or provide pretrained sentence-transformer model
# model = SentenceTransformer("msmarco-distilbert-base-v3")
print(device)
retriever = TrainRetriever(model=model, batch_size=32)

#### Prepare training samples
train_samples = retriever.load_train(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

#### Training SBERT with cosine-product
if args.product == "cosine":
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
    score_functions = {'cos_sim': util.cos_sim}
#### training SBERT with dot-product
elif args.product == "dot":
    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)
    score_functions = {'dot_score': util.dot_score}
#### Prepare dev evaluator
corpus_chunk_size=100000
print("IR evaluation without flush")
print("IR evaluation",flush=True)
ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels, name="dev")

#### If no dev set is present from above use dummy evaluator
# ir_evaluator = retriever.load_dummy_evaluator()

#### Configure Train params
num_epochs = args.num_epochs
# evaluation_steps = math.ceil(len(train_samples)/retriever.batch_size)
# set -1 to evaluate after each epoch
evaluation_steps = -1
warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

print(">>> Starting training now...", flush=True)
print(model_name)
retriever.fit(train_objectives=[(train_dataloader, train_loss)], 
                evaluator=ir_evaluator, 
                epochs=num_epochs,
                output_path=model_save_path,
                warmup_steps=warmup_steps,
                evaluation_steps=evaluation_steps,
                use_amp=True,
                callback=lambda score, epoch, steps: print(f"[Epoch {epoch} | Step {steps}] Eval score: {score}", flush=True)
)
'''



"""
Train a Bi-Encoder (DPR-style) with quality-weighted weak supervision.
Correct implementation for SentenceTransformers.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sentence_transformers import models, SentenceTransformer, InputExample
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
import argparse
from os.path import join, dirname, abspath
import sys

print("Started", flush=True)

############################################
# PATH SETUP
############################################
zhiyuan_path = dirname(dirname(dirname(dirname(abspath(__file__)))))
if zhiyuan_path not in sys.path:
    sys.path.append(zhiyuan_path)

from weak_data_loader import WeakDataLoader

data_dir = join(zhiyuan_path, "datasets")
raw_dir = join(data_dir, "raw")
weak_dir = join(data_dir, "weak")
beir_dir = join(raw_dir, "beir")
xuyang_dir = join(dirname(zhiyuan_path), "xuyang", "data")

############################################
# ARGUMENTS
############################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default="msmarco", type=str)
parser.add_argument('--num_epochs', default=2, type=int)
parser.add_argument('--train_num', default=50, type=int)
parser.add_argument('--weak_num', default="5000", type=str)
parser.add_argument('--product', default="cosine", type=str)
parser.add_argument('--exp_name', default="no_aug", type=str)
args = parser.parse_args()

############################################
# MODEL SAVE PATH
############################################
model_name = "bert-large-uncased"
model_save_path = os.path.join(
    pathlib.Path(__file__).parent.absolute(),
    "output",
    args.exp_name,
    str(args.train_num),
    f"{model_name}-v1-{args.dataset_name}"
)
os.makedirs(model_save_path, exist_ok=True)

############################################
# LOGGING
############################################
fh = logging.FileHandler(join(model_save_path, "log.txt"))
ch = logging.StreamHandler(sys.stdout)
logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[fh, ch]
)

############################################
# LOAD DATA
############################################
if args.exp_name == "no_aug":
    corpus, queries, qrels = GenericDataLoader(
        corpus_file=join(beir_dir, args.dataset_name,
                         f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
        query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
        qrels_file=join(xuyang_dir,
                        f"{args.dataset_name}_{args.train_num}",
                        f"prompt_tuning_{args.train_num}.tsv")
    ).load_custom()
else:
    weak_query_file = join(
        xuyang_dir,
        f"{args.dataset_name}_{args.train_num}",
        args.weak_num,
        f"weak_queries_{args.train_num}_{args.exp_name}.jsonl"
    )
    weak_qrels_file = join(
        xuyang_dir,
        f"{args.dataset_name}_{args.train_num}",
        args.weak_num,
        f"weak_train_{args.train_num}_{args.exp_name}.tsv"
    )

    corpus, queries, qrels = WeakDataLoader(
        corpus_file=join(beir_dir, args.dataset_name,
                         f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
        query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
        qrels_file=join(xuyang_dir,
                        f"{args.dataset_name}_{args.train_num}",
                        f"prompt_tuning_{args.train_num}.tsv"),
        weak_query_file=weak_query_file,
        weak_qrels_file=weak_qrels_file
    ).load_weak_custom()

############################################
# DEV SET
############################################
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(
    corpus_file=join(beir_dir, args.dataset_name,
                     f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
    query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
    qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")
).load_custom()

############################################
# MODEL
############################################
word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer(
    modules=[word_embedding_model, pooling_model],
    device=device
)

retriever = TrainRetriever(model=model, batch_size=32)

############################################
# DATASET WITH QUALITY WEIGHTS
############################################
class WeightedTrainDataset(Dataset):
    def __init__(self, corpus, queries, qrels):
        self.samples = []
        for qid in qrels:
            for pid in qrels[qid]:
                quality = float(qrels[qid][pid])
                self.samples.append(
                    InputExample(
                        texts=[queries[qid], corpus[pid]["text"]],
                        label=quality  # quality_score goes here
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

############################################
# LOAD TRAIN DATA
############################################
train_dataset = WeightedTrainDataset(corpus, queries, qrels)
train_dataloader = retriever.prepare_train(train_dataset, shuffle=True)

############################################
# CORRECT WEIGHTED LOSS
############################################
class WeightedMultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model, similarity_fct=util.cos_sim):
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, sentence_features, labels):
        # Compute embeddings
        embeddings = [
            self.model(sentence_feature)['sentence_embedding']
            for sentence_feature in sentence_features
        ]

        # Similarity matrix
        scores = self.similarity_fct(embeddings[0], embeddings[1])

        # Standard MNR diagonal targets
        target = torch.arange(scores.size(0)).to(scores.device)

        # Per-sample CE loss
        per_sample_loss = self.cross_entropy(scores, target)

        # labels = quality scores from InputExample.label
        if labels is not None:
            weights = labels.float().to(scores.device)

            # Normalize weights for stability
            weights = weights / (weights.mean() + 1e-8)

            loss = (per_sample_loss * weights).mean()
        else:
            loss = per_sample_loss.mean()

        return loss

############################################
# LOSS SELECTION
############################################
if args.product == "cosine":
    train_loss = WeightedMultipleNegativesRankingLoss(model)
else:
    train_loss = WeightedMultipleNegativesRankingLoss(
        model,
        similarity_fct=util.dot_score
    )

############################################
# TRAINING CONFIG
############################################
ir_evaluator = retriever.load_ir_evaluator(
    dev_corpus, dev_queries, dev_qrels, name="dev"
)

num_epochs = args.num_epochs
evaluation_steps = -1
warmup_steps = int(len(train_dataset) * num_epochs / retriever.batch_size * 0.1)

print(">>> Starting training...", flush=True)

retriever.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=ir_evaluator,
    epochs=num_epochs,
    output_path=model_save_path,
    warmup_steps=warmup_steps,
    evaluation_steps=evaluation_steps,
    use_amp=True,
    callback=lambda score, epoch, steps:
        print(f"[Epoch {epoch} | Step {steps}] Eval score: {score}", flush=True)
)
