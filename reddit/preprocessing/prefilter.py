import pandas as pd
import glob
from pathlib import Path
import os
import argparse

# Directory params for download
DATA_PATH = Path('..') / 'data'
RAW_DIR = DATA_PATH / 'raw'
RAW_DIR.mkdir(parents=True, exist_ok=True)
META_DIR = DATA_PATH / 'meta'
META_DIR.mkdir(parents=True, exist_ok=True)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--min-posts', type=int, default=5,
                    help='Minimum number of posts per user')
                    

def prefilter_authors(min_posts=5):
    ''' Indexes and removes all authors with fewer posts than threshold'''
    print(f'Pre-filtering authors...')
    fs = glob.glob(str(RAW_DIR/'*'))
    
    # Find valid users
    count_posts_raw = 0
    for idx, f in enumerate(fs):
        print(f'Reading {idx+1} of {len(fs)}')
        df = pd.read_csv(f, sep='\t', compression='gzip', 
                            lineterminator='\n')
        count_posts_raw += df.shape[0]
        # Count number of posts per author in file
        adf = df[['author']].groupby('author').agg('count')
        adf.columns = adf.columns.droplevel()
        adf = adf.reset_index().rename({'count': 'n_user_posts'}, axis=1)
        # Append to overall df and sum at intervals
        if idx == 0:
            all_adf = adf
        else:
            all_adf = pd.concat([all_adf, adf], 
                                ignore_index=True)
        if idx % 20 == 0:
            all_adf = all_adf.groupby('author', 
                                        as_index=False).sum()
     
    # Number of posts before filtering
    print(f'Number of posts before filtering: {count_posts_raw}')
    
    #  Remove authors below threshold and store info
    all_adf = all_adf.groupby('author', as_index=False).sum()
    all_adf = all_adf[all_adf['n_user_posts']>=min_posts]
    all_adf.to_csv(str(META_DIR/'valid_users.txt.gz'),
                   index=False, sep='\t', compression='gzip')
    
    # Log number of posts/authors so far
    print(f'Number of valid users: {all_adf.shape[0]}')
    
    # Read files again and remove authors with < min_posts posts
    count_posts_filtered = 0
    all_adf = pd.read_csv(str(META_DIR/'valid_users.txt.gz'),
                          sep='\t', compression='gzip', 
                          lineterminator='\n')
    valid_users = all_adf['author'].unique().tolist()
    for f in fs:
        df = pd.read_csv(f, sep='\t', compression='gzip', 
                         lineterminator='\n')
        df = df[df['author'].isin(valid_users)]
        outfile = f.rstrip('.txt.gz') + '_filtered.txt.gz'
        df.to_csv(outfile, sep='\t', compression='gzip', index=False)
        count_posts_filtered += df.shape[0]
        print(f'\tTotal posts at {f}: {count_posts_filtered}')
        os.remove(f) # Remove original file
    
    # Number of valid authors
    print(f'Number of posts after filtering: {count_posts_filtered}')


if __name__=="__main__":
    args = parser.parse_args()
    prefilter_authors(args.num_posts)