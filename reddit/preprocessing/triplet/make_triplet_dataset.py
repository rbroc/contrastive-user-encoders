from reddit.utils import save_tfrecord
from pathlib import Path
import json
import tensorflow as tf
import argparse
import glob
from multiprocessing import Pool
from transformers import AutoTokenizer
import gzip

TRIPLET_PATH = Path('..') / '..' / 'data' / 'json' / 'triplet'
DATASET_PATH = Path('..') / '..' / 'data' / 'datasets' / 'triplet'

parser = argparse.ArgumentParser()
parser.add_argument('--n-cores', type=int, default=10,
                    help='Number of parallel calls to main loop')
parser.add_argument('--tokenizer-weights', type=str, 
                    default='distilbert-base-uncased',
                    help='Weights of pretrained tokenizer')
parser.add_argument('--n-shards', type=int, default=1, 
                    help='Number of shards for each batch of users')
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Dataset name for output path')

def _tknz(x, tokenizer):
    ''' Tokenize single user '''
    out = tokenizer.batch_encode_plus(x,
                                      truncation=True, 
                                      padding='max_length')
    return out


def _generate_example(d, tknzr):
    ''' Generator for TFDataset '''
    for adict in d:
        anchor, pos, neg = [_tknz(adict[e], tknzr) 
                            for e in ['anchor', 'positive', 'negative']]
        ids, pids, nids = [x['input_ids'] for x in [anchor, pos, neg]]
        mask, pmask, nmask = [x['attention_mask'] for x in [anchor, pos, neg]]
        yield (ids, mask, 
               pids, pmask, 
               nids, nmask, 
               adict['author_id'])


def make_dataset(f, outpath, tknzr, n_shards=1):
    ''' Create dataset and save as tfrecord'''
    print(f'Processing {f}...')
    fid = f.split('/')[-1].split('.')[0]
    d = json.load(gzip.open(f))
    ds = tf.data.Dataset.from_generator(lambda: _generate_example(d, tknzr),
                                        output_types=tuple([tf.int32]*7))
    save_tfrecord(ds, fid, outpath, n_shards=n_shards)


if __name__=="__main__":
    args = parser.parse_args()
    OUTPATH = DATASET_PATH / str(args.dataset_name)
    OUTPATH.mkdir(exist_ok=True, parents=True)
    fs = glob.glob(str(TRIPLET_PATH / args.dataset_name / '*'))
    tknzr = AutoTokenizer.from_pretrained(args.tokenizer_weights)
    pool = Pool(processes=args.n_cores)
    pool.starmap(make_dataset, zip(fs,
                                   [OUTPATH]*len(fs),
                                   [tknzr]*len(fs),
                                   [args.n_shards]*len(fs)))
