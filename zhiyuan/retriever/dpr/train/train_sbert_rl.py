'''
Train a Bi-Encoder (DPR-style) using MultipleNegativesRankingLoss.
Optionally performs LISTWISE RL fine-tuning.

Example:
python train_sbert.py --dataset_name msmarco --use_rl --rl_steps 1000
'''

import torch
from sentence_transformers import losses, models, SentenceTransformer
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import pathlib, os
import logging
import argparse
from os.path import join, dirname, abspath
import sys
import random

print("Started", flush=True)

zhiyuan_path = dirname(dirname(dirname(dirname(abspath(__file__)))))
if zhiyuan_path not in sys.path:
    sys.path.append(zhiyuan_path)

from weak_data_loader import WeakDataLoader

def _get_doc_text(doc):
    text = doc.get("text", "")
    title = doc.get("title", "")
    return f"{title} {text}".strip()

def main(args):

    # -------------------------
    # Paths
    # -------------------------
    data_dir = join(zhiyuan_path, "datasets")
    raw_dir = join(data_dir, "raw")
    beir_dir = join(raw_dir, "beir")
    xuyang_dir = join(dirname(zhiyuan_path), "xuyang", "data")

    model_name = "bert-large-uncased"
    model_save_path = os.path.join(
        pathlib.Path(__file__).parent.absolute(),
        "output",
        args.exp_name,
        str(args.train_num),
        f"{model_name}-v1-{args.dataset_name}"
    )
    os.makedirs(model_save_path, exist_ok=True)

    fh = logging.FileHandler(join(model_save_path, "log.txt"))
    ch = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        handlers=[fh, ch]
    )

    # -------------------------
    # Load data
    # -------------------------
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

    dev_corpus, dev_queries, dev_qrels = GenericDataLoader(
        corpus_file=join(beir_dir, args.dataset_name,
                         f"corpus_{args.weak_num}_reduced_ratio_20.jsonl"),
        query_file=join(beir_dir, args.dataset_name, "queries.jsonl"),
        qrels_file=join(beir_dir, args.dataset_name, "qrels", "dev.tsv")
    ).load_custom()

    # -------------------------
    # Model
    # -------------------------
    word_embedding_model = models.Transformer(model_name, max_seq_length=350)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model],
        device=device
    )

    retriever = TrainRetriever(model=model, batch_size=16)

    train_samples = retriever.load_train(corpus, queries, qrels)
    train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

    train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)

    ir_evaluator = retriever.load_ir_evaluator(
        dev_corpus, dev_queries, dev_qrels, name="dev"
    )

    num_epochs = args.num_epochs
    evaluation_steps = -1
    warmup_steps = int(
        len(train_samples) * num_epochs / retriever.batch_size * 0.1
    )

    print(">>> Starting supervised DPR training...", flush=True)
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

    # -------------------------
    # LISTWISE RL Fine-Tuning
    # -------------------------
    if args.use_rl and args.rl_steps > 0:
        print(f">>> Starting LISTWISE RL fine-tuning ({args.rl_steps} steps)", flush=True)
        model.train()

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.rl_lr)
        query_ids = [qid for qid, rels in qrels.items() if len(rels) > 0]
        doc_ids = list(corpus.keys())
        top_k = 50  # top-k candidates per step

        for step in range(args.rl_steps):
            # Sample a query
            qid = random.choice(query_ids)
            query_text = queries[qid]

            # Positive + negatives
            pos_ids = list(qrels[qid].keys())
            neg_ids = [did for did in doc_ids if did not in qrels[qid]]
            candidate_ids = random.sample(pos_ids, min(1, len(pos_ids))) + random.sample(neg_ids, top_k-1)
            random.shuffle(candidate_ids)
            candidate_texts = [_get_doc_text(corpus[did]) for did in candidate_ids]

            # Encode query + candidates
            features = model.tokenize([query_text]+candidate_texts)
            features = {k: v.to(model.device) for k, v in features.items()}
            embeddings = model(features)["sentence_embedding"]

            query_emb = embeddings[0:1]
            doc_embs = embeddings[1:]
            sims = torch.cosine_similarity(query_emb, doc_embs, dim=-1)

            # Sample one document (action) from similarity distribution
            dist = torch.distributions.Categorical(logits=sims)
            action = dist.sample()

            # Reward = 1 if selected doc is relevant, 0 otherwise
            reward = 1.0 if candidate_ids[action.item()] in qrels[qid] else 0.0

            # REINFORCE loss
            loss = -dist.log_prob(action) * reward

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step+1) % 50 == 0 or step == 0:
                print(f"[RL step {step+1}/{args.rl_steps}] reward={reward:.1f}, loss={loss.item():.6f}", flush=True)

        # Save fine-tuned model
        rl_save_path = os.path.join(model_save_path, "rl-listwise")
        os.makedirs(rl_save_path, exist_ok=True)
        model.save(rl_save_path)
        print(f">>> LISTWISE RL fine-tuning complete. Model saved at {rl_save_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default="msmarco", type=str)
    parser.add_argument('--num_epochs', default=2, type=int)
    parser.add_argument('--train_num', default=50, type=int)
    parser.add_argument('--weak_num', default="5000", type=str)
    parser.add_argument('--exp_name', default="no_aug", type=str)
    parser.add_argument('--use_rl', action='store_true', help="Enable LISTWISE RL fine-tuning")
    parser.add_argument('--rl_steps', default=0, type=int)
    parser.add_argument('--rl_lr', default=1e-6, type=float)
    args = parser.parse_args()
    main(args)
