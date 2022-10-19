import tensorflow as tf
from tensorflow.data import TFRecordDataset
from tensorflow.train import Example, Features, Feature, BytesList
from tensorflow.io import FixedLenFeature
import os


FEATURE_NAMES = {'triplet': ['iids', 'amask', 
                             'pos_iids', 'pos_amask',
                             'neg_iids', 'neg_amask', 
                             'author_id'],
                 'triplet_baselines': ['iids', 'labels'],
                 'subreddit_classification': ['iids', 'amask', 'labels']}

FEATURE_OUT_TYPES = {'triplet': tuple([tf.int32]*7),
                     'subreddit_classification': (tf.int32, tf.int32, tf.float32),
                     'triplet_baselines': (tf.int32, tf.float32)}



def _get_feature_description(feature_names):  
    types = [FixedLenFeature([], tf.string)]*len(feature_names)
    return dict(zip(feature_names, types))


def _set_make_example_opts(ds_type):
    def _make_example(*inp):
        feature_values = [Feature(
                            bytes_list=BytesList(value=[
                                tf.io.serialize_tensor(x).numpy(),
                                ])) 
                          for x in inp]
        features = Features(feature=dict(zip(FEATURE_NAMES[ds_type],
                                             feature_values)))
        example = Example(features=features)
        return example.SerializeToString()
    return _make_example


def _make_examples(*inps, ds_type):
    ''' Maps make_example to whole dataset '''
    to_string = tf.py_function(_set_make_example_opts(ds_type),
                               inp=inps, 
                               Tout=tf.string)
    return to_string


def _shard_fn(k, ds, prefix, path, compression, n_shards):
    ''' Util function to shard dataset at save '''
    str2 = tf.strings.join([os.sep, 
                            prefix, 
                            '-',
                            tf.strings.as_string(k), 
                            '-of-', str(n_shards-1), '.tfrecord'])
    fname = tf.strings.join([str(path), str2])    
    writer = tf.data.experimental.TFRecordWriter(fname, 
                                                 compression_type=compression)
    writer.write(ds.map(lambda _, x: x))
    return tf.data.Dataset.from_tensors(fname)


def save_tfrecord(dataset, prefix, path,
                  n_shards=1, 
                  ds_type='triplet',
                  compression='GZIP'):
    ''' Saves tfrecord in shards
        Args:
            dataset (TFDataset): dataset to be saved
            prefix (str): prefix for tfrecord file
            path (str or Path): output path for dataset
            n_shards (int): number of shards (defaults to 1)
            ds_type (str): type of dataset (of those supported in 
                feature names schema)
            compression (str): type of compression to apply
    '''
    
    dataset = dataset.map(lambda *x: _make_examples(*x, 
                                                    ds_type=ds_type)).enumerate()
    dataset = dataset.apply(tf.data.experimental.group_by_window(
                            lambda i, _: i % n_shards, 
                            lambda k, d: _shard_fn(k, d, prefix,
                                                   path,
                                                   compression, 
                                                   n_shards), 
                            tf.int64.max ))
    for s in dataset:
        print(f'Saving {s} from {prefix} ...')


def _parse_fn(example, ds_type='triplet'):
    ''' Parse examples at load '''
    feature_names = FEATURE_NAMES[ds_type]
    feature_types = FEATURE_OUT_TYPES[ds_type]
    example = tf.io.parse_single_example(example, 
                                         _get_feature_description(feature_names))
    inps = [tf.io.parse_tensor(example[f], feature_types[idx]) 
            for idx, f in enumerate(feature_names)]
    return dict(zip(feature_names, inps))


def load_tfrecord(fnames, 
                  num_parallel_calls=1,
                  deterministic=False,
                  cycle_length=16,
                  compression='GZIP',
                  ds_type='triplet'):
    ''' Loads dataset from tfrecord files
        Args:
            fnames (list): list of filenames for TFRecord
            num_parallel_calls (int): number of parallel reads
            deterministic (bool): does order matter (tradeoff with speed)
            cycle_length (int): number of input elements processed concurrently
            compression (str): type of compression of the target files
    '''
    opts = tf.data.Options()
    opts.experimental_deterministic = deterministic
    dataset = tf.data.Dataset.from_tensor_slices(fnames)
    dataset = dataset.with_options(opts)
    read_fn = lambda x: tf.data.TFRecordDataset(x, 
                                                compression_type=compression)
    dataset = dataset.interleave(read_fn, 
                                 cycle_length=cycle_length, 
                                 num_parallel_calls=num_parallel_calls)
    return dataset.map(lambda x: _parse_fn(x, ds_type),
                       num_parallel_calls=num_parallel_calls)
