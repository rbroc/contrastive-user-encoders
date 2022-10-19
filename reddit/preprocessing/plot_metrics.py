import pandas as pd
import numpy as np
import glob
from pathlib import Path
import seaborn as sns
from matplotlib import pyplot as plt


DATA_PATH = Path('..') / 'data'
PROCESSED_PATH = DATA_PATH  /'filtered'
FIG_PATH = DATA_PATH / 'figures'
FIG_PATH.mkdir(exist_ok=True)
META_PATH = DATA_PATH / 'meta'


def compute_metrics():
    ''' Computes and plots dataset metrics after filtering '''
    fs = glob.glob(str(PROCESSED_PATH/'*'))
    for idx, f in enumerate(fs):
        print(f'Reading {f}...')
        df = pd.read_csv(f, sep='\t', compression='gzip')
        udf = df.groupby('author').agg({'selftext': 'count',
                                        'subreddit': pd.Series.nunique})
        udf = udf.reset_index()
        if idx == 0:
            agg_df = udf
        else:
            agg_df = pd.concat([agg_df, udf], 
                               ignore_index=True)
                               
    # Update list of valid authors
    agg_df.to_csv(str(META_PATH/'valid_users.txt.gz'), 
                  sep='\t',
                  compression='gzip', index=False)

    # Compute metrics
    pmean = agg_df['selftext'].mean()
    pmax = agg_df['selftext'].max()
    pmin = agg_df['selftext'].min()
    smean = agg_df['subreddit'].mean()
    smax = agg_df['subreddit'].max()
    smin = agg_df['subreddit'].min()

    # Feedback on metrics
    print(f'Min, avg, max posts per user: {[pmin, pmean, pmax]}')
    print(f'Min, avg, max subreddits per user: {[smin, smean, smax]}')

    # Plot and save aggregates
    figname = str(FIG_PATH/'aggregates.png')
    f, ax = plt.subplots(nrows=2)
    vars = ['selftext', 'subreddit']
    xlabs = ['# posts', '# subreddits']
    for i in range(2):
        sns.histplot(x=agg_df[vars[i]], ax=ax[i], 
                     bins=np.arange(0,1000,10))
        ax[i].set_xlabel(xlabs[i])
        ax[i].set_ylabel('# users')
        ax[i].legend('')
        ax[i].set_xlim(0,500)
    plt.tight_layout()
    plt.savefig(figname)


if __name__=="__main__":
    compute_metrics()