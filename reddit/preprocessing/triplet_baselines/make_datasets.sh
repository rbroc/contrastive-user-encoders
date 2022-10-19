#!/bin/sh
for s in 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30
do
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 100 --mode binary
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 1000 --mode binary
    #python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 5000 --mode binary
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 100 --mode count
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 1000 --mode count
    # python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 5000 --mode count
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 100 --mode tfidf
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 1000 --mode tfidf
    # python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 5000 --mode tfidf
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 100 --mode freq
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 1000 --mode freq
    # python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 5000 --mode freq
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype wav2vec
done

for s in 4 5 6 7 8 9 10 11 12 13 14 15 16 17 19 20 21 22 23 24 25 26 27 28 29 30
do
    #python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 100 --mode binary
    #python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 1000 --mode binary
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 5000 --mode binary
    #python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 100 --mode count
    #python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 1000 --mode count
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 5000 --mode count
    #python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 100 --mode tfidf
    #python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 1000 --mode tfidf
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 5000 --mode tfidf
    #python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 100 --mode freq
    #python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 1000 --mode freq
    python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype bow --tsize 5000 --mode freq
    #python3 make_triplet_baselines_dataset.py --dataset-name single_$s --ttype wav2vec
done
