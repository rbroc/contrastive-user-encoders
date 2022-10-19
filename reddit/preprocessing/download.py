import zstandard as zstd
import lzma
import requests
import os
from pathlib import Path
import json
import itertools
import pandas as pd
import wget
import fasttext

# Directory params for download
DOWNLOAD_DIR = Path('..') / 'data' / 'tmp'
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = Path('..') / 'data' / 'raw'
SAVE_DIR.mkdir(parents=True, exist_ok=True)
FASTTEXT_FILE = DOWNLOAD_DIR / 'lid.176.bin'

# Url for requests
URL = 'https://files.pushshift.io/reddit/submissions/'

# Pushshift files params
years = [str(i) for i in [2018, 2019]]
months = ['0' + str(i) for i in range(1, 10)] + [str(i) for i in [10, 11, 12]]
ym_comb = itertools.product(years, months)

# Filtering params
target_fields = ['author', 'created_utc', 'id', 
                'num_comments', 'score', 'selftext', 
                'subreddit', 'title']

# Language detection model
FASTTEXT_URL = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
if not os.path.exists(FASTTEXT_FILE):
    print(f'Downloading FastText language detection model...')
    wget.download(FASTTEXT_URL, out=str(FASTTEXT_FILE))
langdetect = fasttext.load_model(str(FASTTEXT_FILE))

# Define language filtering function
def _language_detection(s):
    try:
        return langdetect.predict(s)[0][0].split('__')[2]
    except:
        return 'unk'

# Filter submissions by content and do minimal preprocessing
def filter_submission(ldict, posts, target_fields):
    if (ldict['over_18'] is False) and \
        (ldict['selftext'] not in ['', '[deleted]', '[removed]']) and \
        (ldict['author'] != '[deleted]') and \
        (ldict['is_self'] == True) and \
        (ldict['num_crossposts'] == 0):
        ldict = {k: ldict[k] for k in target_fields}
        posts.append(ldict)
    return posts

# Define dataframe-wide filtering function
def clean_df(df):
    df = df.drop_duplicates(subset=['author', 'selftext'])
    df = df.dropna(subset=['author','selftext','subreddit'])
    df['selftext'].replace('[\s]+', ' ', regex=True, inplace=True)
    df['selftext'].replace('\"', '', regex=True, inplace=True)
    df['lang'] = df['selftext'].apply(_language_detection)
    df = df[df['lang']=='en'].drop('lang', axis=1)
    return df

# Define logging function
def save_file(posts, year, month, idx, fprefix):
    idx += 1
    df = clean_df(pd.DataFrame(posts))
    outfile = SAVE_DIR / f'{year}_{month}_{idx}.txt.gz'
    df.to_csv(outfile, sep='\t', index=False, compression='gzip', line_terminator='\n')
    print(f'Saving {fprefix} {idx}...')
    posts = []
    return posts, idx


# Main function
def download_and_extract():
    ''' Downloads Reddit dump, filters posts and saves as tsv '''
    for year, month in ym_comb:
        # Request
        fprefix = ''.join(['RS', '_', year, '-', month])
        furl = ''.join([URL, fprefix])
        cformat = '.zst'
        r = requests.get(furl + cformat, stream=True)
        if r.status_code == 404:
            cformat = '.xz'
            r = requests.get(furl + '.xz', stream=True)

        # Download and save
        if not os.path.exists(DOWNLOAD_DIR/(fprefix+cformat)):
            with open(DOWNLOAD_DIR/(fprefix+cformat), 'wb') as f:
                for idx, chunk in enumerate(r.iter_content(chunk_size=16384)):
                    if (idx != 0) and (idx % 1000 == 0):
                        print(f'Writing file {(fprefix + cformat)}: chunk {idx}')
                    f.write(chunk)
                print('Done writing!')
        else:
            print(f'{fprefix} already downloaded!')

        # Decompress filter and save file
        posts = []
        idx = 0

        # xz format 
        if cformat == '.xz':
            with lzma.open(DOWNLOAD_DIR/(fprefix+cformat), mode='rt') as fh:
                for line in fh:
                    ldict = json.loads(line)
                    posts = filter_submission(ldict, posts, target_fields)
                    if len(posts) == 100000:
                        posts, idx = save_file(posts, year, month, idx, fprefix)
                if posts != []:
                    idx += 1
                    outfile = SAVE_DIR / f'{year}_{month}_{idx}.txt.gz'
                    df = clean_df(pd.DataFrame(posts))
                    df.to_csv(outfile, sep='\t', index=False, 
                              line_terminator='\n',
                              compression='gzip')

        # zst format
        elif cformat == '.zst':
            with open(DOWNLOAD_DIR/(fprefix+cformat), 'rb') as fh:
                dcmp = zstd.ZstdDecompressor()
                buffer = ''
                with dcmp.stream_reader(fh) as reader:
                    while True:
                        chunk = reader.read(8192).decode()
                        if not chunk:
                            if posts != []:
                                idx += 1
                                outfile = SAVE_DIR / f'{year}_{month}_{idx}.txt.gz'
                                df = clean_df(pd.DataFrame(posts))
                                df.to_csv(outfile, sep='\t', index=False, 
                                          compression='gzip',
                                          line_terminator='\n')
                            break
                        lines = (buffer + chunk).split('\n')
                        for line in lines[:-1]:
                            ldict = json.loads(line)
                            posts = filter_submission(ldict, posts, target_fields)
                            if len(posts) == 100000:
                                posts, idx = save_file(posts, year, month, idx, fprefix)
                        buffer = lines[-1]          
    
        os.remove(DOWNLOAD_DIR/(fprefix+cformat))
    os.remove(FASTTEXT_FILE)
    os.rmdir(DOWNLOAD_DIR)


if __name__ == '__main__':
    download_and_extract()