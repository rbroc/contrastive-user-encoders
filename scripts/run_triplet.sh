#!/bin/sh

#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path final_1anchor --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pooling cls --pretrained-weights distilbert-base-uncased --filter-by-n

#python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path final_10anchor --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pooling cls --pretrained-weights distilbert-base-uncased --filter-by-n

python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path final_1anchor_baseline_validation --per-replica-batch-size 1 --n-epochs 1 --update-every 1 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pooling cls --pretrained-weights distilbert-base-uncased --test-only

python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path final_1anchor_baseline_test --per-replica-batch-size 1 --n-epochs 1 --update-every 1 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pooling cls --pretrained-weights distilbert-base-uncased --test-only --which-set test

python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path final_1anchor_test --per-replica-batch-size 1 --n-epochs 1 --update-every 1 --loss-margin 1 --pad-anchor 10 --n-anchor 1 --n-pos 1 --n-neg 1 --pooling cls --pretrained-weights ../trained/final_1anchor/epoch-0 --test-only --which-set test

python3 train_triplet.py --dataset-name 1pos_1neg_random --log-path final_10anchor_test --per-replica-batch-size 1 --n-epochs 1 --update-every 1 --loss-margin 1 --pad-anchor 10 --n-anchor 10 --n-pos 1 --n-neg 1 --pooling cls --pretrained-weights ../trained/final_10anchor/epoch-0 --test-only --which-set test
