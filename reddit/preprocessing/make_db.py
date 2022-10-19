import sqlite3
import glob
from pathlib import Path
import pandas as pd

DATA_PATH = Path('..') / 'data'
FILTERED_PATH = DATA_PATH / 'filtered' # could do on raw
DB_PATH = DATA_PATH / 'databases'
fs = glob.glob(str(FILTERED_PATH/'*'))


def create_db():
    conn = sqlite3.connect(str(DB_PATH / 'posts.db'))
    cur = conn.cursor()

    # Create the table
    cur.execute(""" CREATE TABLE posts (
                    id text,
                    author text,
                    created_utc integer,
                    num_comments integer,
                    score integer,
                    selftext text,
                    subreddit text,
                    title text
                    )""")
    conn.commit()

    # Read files and append to db
    for idx, f in enumerate(fs):
        print(f'Appending {idx} of {len(fs)}')
        df = pd.read_csv(f, sep='\t', compression='gzip')
        df.to_sql('posts', conn, 
                  if_exists='append', 
                  index_label='id', 
                  index=False)
        conn.commit()
        
    # Create indices for author and subreddit
    cur.execute('''CREATE INDEX author_idx 
                   ON posts (author)''')
    conn.commit()
    cur.execute('''CREATE INDEX subreddit_idx 
                   ON posts (subreddit)''')
    conn.commit()
    
    # Get summary metrics
    cur.execute("SELECT COUNT(selftext) FROM posts")
    print(f'Nr posts: {cur.fetchall()}')
    
    cur.execute("SELECT COUNT(DISTINCT(author)) FROM posts")
    print(f'Nr authors: {cur.fetchall()}')

    cur.execute("SELECT COUNT(DISTINCT(subreddit)) FROM posts")
    print(f'Nr authors: {cur.fetchall()}')

    conn.close()
    

if __name__=='__main__':
    create_db()