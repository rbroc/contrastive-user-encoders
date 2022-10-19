#!/bin/sh

for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30
do
    for posts in 10
    do
        python3 train_subreddits.py --dataset-name single_$s --log-path final_1anchor_epoch0_$posts --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights ../trained/final_1anchor/epoch-0 --target-dims 1 --nr $posts --pad-to $posts
    done
done

for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30
do
    for posts in 10
    do  
        python3 train_subreddits.py --dataset-name single_$s --log-path final_10anchor_epoch0_$posts --per-replica-batch-size 1 --n-epochs 3 --update-every 1 --weights ../trained/final_10anchor/epoch-0 --target-dims 1 --nr $posts --pad-to $posts
    done
done


# 1 Layer
for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30
do
    for d in 100 1000
    do
        for t in binary count freq tfidf
        do
            python3 train_triplet_baselines.py --dataset-name single_${s}_bow_${d}_${t} --log-path 1layer_10 --pad-to 10 --nr 10 --n-layers 1 --input-size $d --which avg
        done
    done 
done

for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30
do
        python3 train_triplet_baselines.py --dataset-name single_${s}_wav2vec --log-path 1layer_10 --pad-to 10 --nr 10 --n-layers 1 --input-size 300 --which avg
done

# 2 Layers
for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30
do
    for d in 100 1000
    do
        for t in binary count freq tfidf
        do
            python3 train_triplet_baselines.py --dataset-name single_${s}_bow_${d}_${t} --log-path 2layers_10 --pad-to 10 --nr 10 --n-layers 2 --input-size $d --which avg
        done
    done 
done

for s in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30
do
        python3 train_triplet_baselines.py --dataset-name single_${s}_wav2vec --log-path 2layers_10 --pad-to 10 --nr 10 --n-layers 2 --input-size 300 --which avg
done