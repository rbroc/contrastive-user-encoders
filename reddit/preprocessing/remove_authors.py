import pandas as pd
from pathlib import Path
import argparse
import csv
import glob
import os

# Define paths
DATA_PATH = Path('..') / 'data'
AUTHOR_PATH = DATA_PATH / 'users'
AUTHOR_PATH.mkdir(exist_ok=True)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--min-posts', type=int, default=5,
                    help='Minimum number of posts per user')
parser.add_argument('--min-subreddits', type=int, default=5,
                    help='Minimum number of subreddits per user')

def remove_authors(min_posts=5, min_subreddits=5):
    ''' Removes single-author files with fewer posts than threshold
    Args:
        min_posts (int): minimum number of posts per user
        min_subreddits (int): mininum number of subreddits to 
            which the user has to have contributed    
    '''
    afs = glob.glob(str(AUTHOR_PATH/'*'))
    idx = 0
    for f in afs:
        idx += 1
        # Read file
        adf = pd.read_csv(f, sep='\t', header=None,
                          quoting=csv.QUOTE_NONE)

        # Remove duplicates and count posts/subreddits
        adf = adf.drop_duplicates(subset=[5])
        # adf = adf[adf[5].str.len()<20]
        nps = adf.shape[0]
        nss =  adf[6].nunique()

        # Remove if does not fit the threshold
        if (nps < min_posts) or (nss < min_subreddits):
            os.remove(f)
        else:
            adf.to_csv(f, sep='\t', index=False)
        
        # Verbose
        if (idx) % 10000 == 0:
            print(f'Processing {idx}')

if __name__ == '__main__':
    args = parser.parse_args()
    remove_authors(args.min_posts, args.min_subreddits)