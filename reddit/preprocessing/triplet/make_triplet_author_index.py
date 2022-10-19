from pathlib import Path
import glob
import gzip
import json

DATA_PATH = Path('..') / '..' / 'data'
META_PATH = DATA_PATH / 'meta'
TRIPLET_PATH = DATA_PATH / 'triplet' / 'json'
outfile = META_PATH / 'triplet_author_index.json.gz'

def make_author_index():
    fs = glob.glob(str(TRIPLET_PATH/'*'))
    adict = {}
    for f in fs:
        fid = f.split('/')[-1].strip('.json.gz')
        bdict = json.load(gzip.open(f))
        alist = [(di['author_id'],
                  di['n_anchor'],
                  di['n_positive'],
                  di['n_negative']) for di in bdict]  
        bindex = {a[0]: {'n_anchor': a[1], 
                         'n_positive': a[2], 
                         'n_negative': a[3],
                         'batch': fid} for a in alist}
        adict.update(bindex)
    with gzip.open(outfile, 'w') as fh:
        fh.write(json.dumps(adict).encode('utf-8'))

if __name__=='__main__':
    make_author_index()