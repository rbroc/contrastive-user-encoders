import pandas as pd
import os
from pathlib import Path
import argparse
import glob
import csv
import gzip
import random
from selectolax.parser import HTMLParser
from markdown2 import markdown
from pandarallel import pandarallel
import sys
import math

csv.field_size_limit(sys.maxsize)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument('--num-parallel-calls', type=int, default=3,
                    help='Number of parallel calls for pandas apply')
parser.add_argument('--batch-size', type=int, default=10000,
                    help='Number of authors per output file')

# Define paths
DATA_PATH = Path('..') / 'data'
PROCESSED_PATH = DATA_PATH / 'filtered'
PROCESSED_PATH.mkdir(exist_ok=True)
AUTHOR_PATH = DATA_PATH / 'users'
AUTHOR_PATH.mkdir(exist_ok=True)

# Util function to remove markdown syntax
def markdown_to_text(md):
    html = markdown(md)
    tree = HTMLParser(html)
    return tree.body.text()

def batch_authors(num_parallel_calls=3, batch_size=10000):
    ''' Batches single-files and removes markdown syntax 
    Args:
        num_parallel_calls (int): number of parallel calls for pandas apply
        batch_size (int): number of users per output file
    '''
    pandarallel.initialize(nb_workers=num_parallel_calls)

    # Filter author files
    afs = glob.glob(str(AUTHOR_PATH/'*'))
    print('Shuffling user order...')
    random.seed(0)
    random.shuffle(afs)
    print('Batching...')
    nfiles = len(afs)
    idx = 0
    for f in afs:
        idx += 1
        if idx % batch_size == 0:
            print(f'\tFile {idx} of {nfiles}...')
        fid = math.ceil(idx/batch_size)
        of = PROCESSED_PATH / f'batch_{fid}.txt.gz'
        with open(f, 'rt') as ifh, gzip.open(of, 'at') as ofh:
            ofh.writelines(ifh.readlines())
        os.remove(f)

    # Re-read files and strip md
    ffs = glob.glob(str(PROCESSED_PATH/'*'))
    print('Stripping markdown syntax...')
    for idx, f in enumerate(ffs):
        print(f'\t{idx} (out of {len(ffs)})')
        df = pd.read_csv(f, sep='\t', compression='gzip', 
                         header=None, quoting=csv.QUOTE_NONE)
        df.columns =  ['author', 'created_utc', 'id', 
                       'num_comments', 'score', 'selftext',
                       'subreddit', 'title']
        df['selftext'] = df['selftext'].astype(str).parallel_apply(markdown_to_text)
        df.to_csv(f, sep='\t', compression='gzip', index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    batch_authors(args.num_parallel_calls, args.batch_size)