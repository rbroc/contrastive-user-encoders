import pandas as pd
import numpy as np
import os
from pathlib import Path
import glob
import csv
import gzip
import sys

csv.field_size_limit(sys.maxsize)

# Define paths
DATA_PATH = Path('..') / 'data'
RAW_PATH = DATA_PATH / 'raw'
AUTHOR_PATH = DATA_PATH / 'users'
AUTHOR_PATH.mkdir(exist_ok=True)


def split_authors():
    ''' Splits files into single-author files '''
    fs = glob.glob(str(RAW_PATH/'*'))
    
    # Split global files into one file per user
    print(f'Splitting files into single-author files...')
    dropped = 0
    for fidx, f in enumerate(fs):
        print(f'{fidx+1} of {len(fs)}')
        with gzip.open(f, 'rt') as ifile:
            rdr = csv.reader(ifile, delimiter='\t')
            for row in rdr:
                if len(row) == 8:
                    ofile = AUTHOR_PATH / f'{row[0]}.txt' 
                    with open(ofile, 'a') as ofh:
                        ofh.write('\t'.join(row)+'\n')
                else:
                    dropped += 1
        print(f'\t{dropped} invalid rows so far')
        os.remove(f)
    os.rmdir(RAW_PATH)

if __name__=='__main__':
    split_authors()

