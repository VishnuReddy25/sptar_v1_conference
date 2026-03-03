#!/bin/bash
# run_iterative.sh
# Runs the full iterative retriever-filter loop

DATASET=$1   # fiqa or msmarco
ROUNDS=${2:-3}  # number of rounds, default 3
BASE_EXP="llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70"

echo "=== Round 0: Train initial SPTAR retriever (BM25 filtered) ==="
python zhiyuan/dpr_eval.py \
    --dataset_name $DATASET \
    --version v1 \
    --gpu_id 0 \
    --train_num 50 \
    -exps ${BASE_EXP}_round0 \
    --weak_num 100k
# This gives you R0

for ROUND in $(seq 1 $ROUNDS); do
    echo ""
    echo "=== Round $ROUND: Retriever-guided filtering + retraining ==="
    
    python zhiyuan/dpr_eval.py \
        --dataset_name $DATASET \
        --version v1 \
        --gpu_id 0 \
        --train_num 50 \
        -exps ${BASE_EXP} \
        --weak_num 100k \
        --use_retriever_filter \
        --iter_round $ROUND
    
    echo "Round $ROUND complete."
done

echo ""
echo "=== All rounds complete. Results in zhiyuan/retriever/dpr/train/output/ ==="
