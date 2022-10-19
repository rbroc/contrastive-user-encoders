import glob
import random
from pathlib import Path
from shutil import copyfile
import json
import argparse
import gzip

META_PATH = Path('..') / '..' / '..' / 'data' / 'meta'
TRIPLET_DS_PATH = Path('..') / '..' / '..' / 'data' / 'datasets' / 'triplet'
TRIPLET_JSON_PATH = Path('..') / '..' / '..' / 'data' / 'json' / 'triplet'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str)

def split(dataset_name):

    # Define paths
    DATASET_PATH = TRIPLET_DS_PATH / dataset_name
    JSON_PATH = TRIPLET_JSON_PATH / dataset_name
    TRAIN_PATH = DATASET_PATH / 'train'
    TEST_PATH = DATASET_PATH / 'test'
    TRAIN_JSON_PATH = JSON_PATH / 'train'
    TEST_JSON_PATH = JSON_PATH / 'test'
    TRAIN_PATH.mkdir(parents=True, exist_ok=True)
    TEST_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_JSON_PATH.mkdir(parents=True, exist_ok=True)
    TEST_JSON_PATH.mkdir(parents=True, exist_ok=True)

    # Get ids
    fs = glob.glob(str(DATASET_PATH / 'batch*'))
    test_fs = list(random.sample(fs, int(len(fs)*.1)))
    train_fs = list(set(fs) - set(test_fs))
    train_nrs = [f.split('/')[-1].split('_')[1].split('-')[0]
                 for f in train_fs]
    test_nrs = [f.split('/')[-1].split('_')[1].split('-')[0]
                for f in test_fs]

    # Move dataset files
    for i, n in enumerate(test_nrs):
        inpj = str(JSON_PATH / f'batch_{n}.json.gz')
        inpd = str(DATASET_PATH / f'batch_{n}-0-of-0.tfrecord')
        outj = str(TEST_JSON_PATH / f'batch_{i}.json.gz') 
        outd = str(TEST_PATH / f'batch_{i}-0-of-0.tfrecord')
        copyfile(inpj, outj)
        copyfile(inpd, outd)
    for i, n in enumerate(train_nrs):
        inpj = str(JSON_PATH / f'batch_{n}.json.gz')
        inpd = str(DATASET_PATH / f'batch_{n}-0-of-0.tfrecord')
        outj = str(TRAIN_JSON_PATH / f'batch_{i}.json.gz') 
        outd = str(TRAIN_PATH / f'batch_{i}-0-of-0.tfrecord')
        copyfile(inpj, outj)
        copyfile(inpd, outd)

    # Now store ids for future reference
    train_json_fs = glob.glob(str(TRAIN_JSON_PATH / '*'))
    test_json_fs = glob.glob(str(TEST_JSON_PATH / '*'))
    train_ids = []
    test_ids = []
    for t in train_json_fs:
        d = json.load(gzip.open(t))
        train_ids += [i['author_id'] for i in d]
    for t in test_json_fs:
        d = json.load(gzip.open(t))
        test_ids += [i['author_id'] for i in d]
    id_dict = {'train_ids': train_ids, 'test_ids': test_ids}
    id_file = str(META_PATH / dataset_name) + '_triplet_splits.json.gz'
    with gzip.open(id_file, 'w') as fh:
        fh.write(json.dumps(id_dict).encode('utf-8'))

if __name__=='__main__':
    args = parser.parse_args()
    split(args.dataset_name)
