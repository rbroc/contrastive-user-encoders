from reddit.utils import save_tfrecord
from pathlib import Path
import json
import tensorflow as tf
import argparse
import glob
from multiprocessing import Pool
import gzip
import os
import spacy
import pickle as pkl

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

TRIPLET_PATH = Path('..') / '..' / 'data' / 'json' / 'subreddit_classification'
DATASET_PATH = Path('..') / '..' / 'data' / 'datasets' / 'triplet_baselines'

parser = argparse.ArgumentParser()
parser.add_argument('--n-cores', type=int, default=20,
                    help='Number of parallel calls to main loop')
parser.add_argument('--ttype', type=str, default=None, 
                    help='bow or wav2vec')
parser.add_argument('--tsize', type=int, default=None, 
                    help='100, 1000, 5000')
parser.add_argument('--mode', type=str, help='tfidf, count, freq, binary')
parser.add_argument('--n-shards', type=int, default=1, 
                    help='Number of shards for each batch of users')
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Dataset name for output path')

def _generate_example(d, tknzr, ttype, mode):
    ''' Generator for TFDataset 
    Args:
        d (list): list of example dictionaries to generate 
                  examples from 
        tknzr (transformers.Tokenizer): Tokenizer
    '''
    for di in d:
        if ttype == 'bow':
            posts = tknzr.texts_to_matrix(di['post'], mode=mode)
        else:
            posts = [tknzr(i).vector for i in di['post']]
        iids = posts
        labels = di['labels']
        yield (iids, labels)

def make_dataset(f, outpath, tknzr, ttype, mode, n_shards=1):
    ''' Create dataset and save as tfrecord'''
    print(f'Processing {f}...')
    fid = f.split('/')[-1].split('.')[0]
    d = json.load(gzip.open(f))
    ds = tf.data.Dataset.from_generator(lambda: _generate_example(d, tknzr, ttype, mode),
                                        output_types=(tf.int32, tf.float32))
    save_tfrecord(ds, fid, outpath, n_shards=n_shards, ds_type='triplet_baselines')


if __name__=="__main__":
    args = parser.parse_args()
    if args.ttype == 'bow':
        tknzr = pkl.load(open(f'tokenizer_{args.tsize}.pkl', 'rb'))
    else:
        tknzr = spacy.load('en_core_web_md')
    for split in ['train', 'val', 'test']:
        if args.ttype == 'bow':
            dname = f'{str(args.dataset_name)}_{args.ttype}_{args.tsize}_{args.mode}'
        else:
            dname = f'{str(args.dataset_name)}_wav2vec'
        OUTPATH = DATASET_PATH / dname / split
        OUTPATH.mkdir(exist_ok=True, parents=True)
        fs = glob.glob(str(TRIPLET_PATH / args.dataset_name / split / '*'))
        pool = Pool(processes=args.n_cores)
        pool.starmap(make_dataset, zip(fs,
                                       [OUTPATH]*len(fs),
                                       [tknzr]*len(fs),
                                       [args.ttype]*len(fs),
                                       [args.mode]*len(fs),
                                       [args.n_shards]*len(fs)))
    pool.close()
        
