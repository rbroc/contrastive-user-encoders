import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json
import glob
import gzip
import random
import itertools
from multiprocessing import Pool

DATA_PATH = Path('..') / 'data'
PROCESSED_PATH = DATA_PATH  / 'filtered'
TRIPLET_PATH = DATA_PATH / 'json' / 'triplet'

parser = argparse.ArgumentParser()
parser.add_argument('--n-cores', type=int, default=10,
                    help='Number of parallel calls to main loop')
parser.add_argument('--seed', type=int, default=0,
                    help='Seed for random call')
parser.add_argument('--n-examples', type=int, default=1,
                    help='Number of negative/positive examples')
parser.add_argument('--batch-size', type=int, default=10000,
                    help='Number of authors per file')
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Dataset name for output path')

def _shuffle_by_author(df):
    groups = [df for _,df in df.groupby('author')]
    random.shuffle(groups)
    shuffled = pd.concat(groups).reset_index(drop=True)
    return shuffled


def pick_and_drop_examples(df, n):
    ''' Picks examples, pops them from main df, returns them separate df 
    Args:
        df (pd.DataFrame): global dataframe
        n (int): number of examples per user 
    '''
    idx = df.groupby('author')\
        .apply(lambda x: np.random.choice(x.index, n, 
                                          replace=False)).values
    idx = list(itertools.chain(*idx))
    ex_df = _shuffle_by_author(df.loc[idx,:])
    df.drop(idx, axis=0, inplace=True)
    return ex_df, df


def _make_user_dict(df, pos_df, neg_df, batch_nr, batch_size):
    ''' Returns dictionary with info on user '''
    alist = []
    for u_idx, u in enumerate(df.author.unique()):
        author_id = (batch_nr - 1) * batch_size + (u_idx + 1)
        anchor = df[df['author']==u]
        neg = neg_df[neg_df['target_author']==u]
        pos = pos_df[pos_df['author']==u]
        adict = {'author': u,
                 'author_id': author_id,
                 'anchor': anchor['selftext'].tolist(),
                 'positive': pos['selftext'].tolist(),
                 'negative': neg['selftext'].tolist(),
                 'n_anchor': anchor.shape[0],
                 'n_positive': pos.shape[0],
                 'n_negative': neg.shape[0],
                 'anchor_subreddits': anchor['subreddit'].tolist(),
                 'positive_subreddits': pos['subreddit'].tolist(),
                 'negative_subreddits': neg['subreddit'].tolist(),
                 'negative_authors': neg['author'].tolist(),
                 'pos_subreddit_overlap': len(set(pos['subreddit']) \
                                              & set(anchor['subreddit'])) / \
                                                  pos.shape[0],
                 'neg_subreddit_overlap': len(set(neg['subreddit']) \
                                              & set(anchor['subreddit'])) / \
                                                  neg.shape[0]}
        alist.append(adict) 
    return alist


# Main function
def make_triplets(f, outpath, seed=0, n_examples=1, 
                  batch_size=10000):
    ''' For each user, selects which posts are used as anchor,
        positive example, and negative example. Stores this info 
        in dataframes (splits into several chunks for ease of 
        processing and storage reasons) and in a json file
    Args:
        seed (int): seed for np.random.seed call
    '''
    print(f'Reading {f}...')
    df = pd.read_csv(f, sep='\t', compression='gzip')
    count_df = df.groupby('author')['selftext'].count().reset_index()
    count_df = count_df.rename({'selftext':'n_posts'}, axis=1)
    df = df.merge(count_df)
    df = df[df['n_posts']>=n_examples*3]
    df['selftext'] = df['selftext'].str.strip()

    # Get examples
    np.random.seed(seed)
    neg_df, df = pick_and_drop_examples(df, n=n_examples)
    pos_df, df = pick_and_drop_examples(df, n=n_examples)
    attempt = 0

    # Match users and examples
    while True: 
        attempt += 1
        print(f'\tMatch users and examples, attempt {attempt}...')
        alist = df.author.unique().tolist()
        np.random.shuffle(alist)
        alist = [a for a in alist for _ in range(n_examples)]
        if all(alist != neg_df['author']):
            break
    neg_df['target_author'] = alist

    # Concatenate and save as json
    ofile_id = f.split('/')[-1].split('.')[0]
    batch_nr = int(ofile_id.split('_')[1])
    ofile_json = outpath / f'{ofile_id}.json.gz'
    d = _make_user_dict(df, pos_df, neg_df, batch_nr, batch_size)
    with gzip.open(ofile_json, 'w') as fh:
        fh.write(json.dumps(d).encode('utf-8'))
        

if __name__=="__main__":
    fs = glob.glob(str(PROCESSED_PATH/'*'))
    args = parser.parse_args()
    OUT_PATH = TRIPLET_PATH / str(args.dataset_name) 
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    pool = Pool(processes=args.n_cores)
    pool.starmap(make_triplets, zip(fs,
                                    [OUT_PATH]*len(fs),
                                    [args.seed]*len(fs),
                                    [args.n_examples]*len(fs),
                                    [args.batch_size]*len(fs)))