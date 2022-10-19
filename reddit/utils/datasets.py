import tensorflow as tf


def filter_triplet_by_n_anchors(x, n_anchor):
    ''' Remove stuff that has too few anchors '''
    if n_anchor:
        n_anch = tf.math.count_nonzero(tf.reduce_sum(x['iids'], 
                                                     axis=1))
        return tf.math.greater(n_anch, n_anchor-1)
    else:
        return True


def pad_and_stack_triplet(dataset, 
                          pad_to=[20,1,1], 
                          n_anchor=None,
                          filter_by_n=True):
    ''' Pads the dataset according to specified number of posts 
        passed via pad_to (anchor, positive, negative) and stacks
        negative, positive and anchor posts vertically.
        Shuffles the anchor comments
    Args:
        dataset (TFDataset): dataset to pad and stack 
        pad_to (list or tuple): list containing number of posts
            to pad to, i.e., [n_anchor_posts, n_positive_posts,
            n_negative_posts]
        n_anchor (int): number of posts kept
        filter_by_n (bool): whether to keep only users that have at least 
            n_anchor posts
    '''
    # Slice
    dataset = dataset.map(lambda x: {'iids': x['iids'][:pad_to[0],:], 
                                     'amask': x['amask'][:pad_to[0],:],
                                     'pos_iids': x['pos_iids'][:pad_to[1],:],
                                     'pos_amask': x['pos_amask'][:pad_to[1],:],
                                     'neg_iids': x['neg_iids'][:pad_to[2],:],
                                     'neg_amask': x['neg_amask'][:pad_to[2],:],
                                     'author_id': x['author_id']})
    # Pad
    dataset = dataset.map(lambda x: {'iids': tf.pad(x['iids'], 
                                                    [[0,pad_to[0]-tf.shape(x['iids'])[0]],
                                                     [0,0]]),
                                     'amask': tf.pad(x['amask'], 
                                                     [[0,pad_to[0]-tf.shape(x['amask'])[0]],
                                                      [0,0]]), 
                                     'pos_iids': tf.pad(x['pos_iids'], 
                                                        [[0,pad_to[1]-tf.shape(x['pos_iids'])[0]],
                                                         [0,0]]),
                                     'pos_amask': tf.pad(x['pos_amask'], 
                                                         [[0,pad_to[1]-tf.shape(x['pos_amask'])[0]],
                                                          [0,0]]), 
                                     'neg_iids': tf.pad(x['neg_iids'], 
                                                        [[0,pad_to[2]-tf.shape(x['neg_iids'])[0]],
                                                         [0,0]]),
                                     'neg_amask': tf.pad(x['neg_amask'], 
                                                         [[0,pad_to[2]-tf.shape(x['neg_amask'])[0]],
                                                          [0,0]]),
                                     'author_id': x['author_id']})
    if n_anchor:
        if filter_by_n:
            dataset = dataset.filter(lambda x: filter_triplet_by_n_anchors(x, 
                                                                           n_anchor))
        dataset = dataset.map(lambda x: {'iids': x['iids'][:n_anchor,:], 
                                         'amask': x['amask'][:n_anchor,:],
                                         'pos_iids': x['pos_iids'],
                                         'pos_amask': x['pos_amask'],
                                         'neg_iids': x['neg_iids'],
                                         'neg_amask': x['neg_amask'],
                                         'author_id': x['author_id']})
    
    # Stack
    dataset = dataset.map(lambda x: {'input_ids': tf.concat([x['neg_iids'],
                                                             x['pos_iids'],
                                                             x['iids']], axis=0),
                                     'attention_mask': tf.concat([x['neg_amask'],
                                                                  x['pos_amask'],
                                                                  x['amask']], axis=0),
                                     'id': x['author_id']})
    return dataset


def _filter_classification(x, n_posts):
    ''' Remove stuff that has too few anchors '''
    n_p1 = tf.math.count_nonzero(tf.reduce_sum(x['iids'], 
                                               axis=1))
    return tf.math.greater(n_p1, n_posts-1)


def stack_classification(dataset, n_posts):
    ''' Stacks examples for classification layer
    Args:
        dataset (TFDataset): dataset to pad and stack 
    '''
    dataset = dataset.filter(lambda x: _filter_classification(x, n_posts)) 
    dataset = dataset.map(lambda x: {'input_ids': tf.concat([x['iids'],
                                                             x['iids2']], axis=0),
                                     'attention_mask': tf.concat([x['amask'],
                                                                  x['amask2']], axis=0),
                                     'labels': x['labels'],
                                     'id': x['author_id']})
    return dataset

def pad_subreddits(dataset, pad_to=3, nr=1):
    ''' Pads the dataset according to specified number of posts 
        passed via pad_to (anchor, positive, negative) and stacks
        negative, positive and anchor posts vertically.
        Shuffles the anchor comments
    Args:
        dataset (TFDataset): dataset to pad and stack 
        pad_to (int): pad to number of posts
        nr (int): number of posts
    '''
    # Slice
    dataset = dataset.map(lambda x: {'iids': x['iids'][:pad_to,:], 
                                     'amask': x['amask'][:pad_to,:],
                                     'labels': x['labels']})
    # Pad
    dataset = dataset.map(lambda x: {'iids': tf.pad(x['iids'], 
                                                    [[0,pad_to-tf.shape(x['iids'])[0]],
                                                     [0,0]]),
                                     'amask': tf.pad(x['amask'], 
                                                     [[0,pad_to-tf.shape(x['amask'])[0]],
                                                      [0,0]]), 
                                     'labels': x['labels']})
    if nr:
        dataset = dataset.map(lambda x: {'input_ids': x['iids'][:nr,:], 
                                         'attention_mask': x['amask'][:nr,:],
                                         'labels': x['labels']})
    return dataset

  
def pad_triplet_baselines(dataset, pad_to=3, nr=1):
    ''' Pads the dataset according to specified number of posts 
        passed via pad_to (anchor, positive, negative) and stacks
        negative, positive and anchor posts vertically.
        Shuffles the anchor comments
    Args:
        dataset (TFDataset): dataset to pad and stack 
        pad_to (int): pad to number of posts
        nr (int): number of posts
    '''
    # Slice
    dataset = dataset.map(lambda x: {'iids': x['iids'][:pad_to,:], 
                                     'labels': x['labels']})
    # Pad
    dataset = dataset.map(lambda x: {'iids': tf.pad(x['iids'], 
                                                    [[0,pad_to-tf.shape(x['iids'])[0]],
                                                     [0,0]]),
                                     'labels': x['labels']})
    if nr:
        dataset = dataset.map(lambda x: {'input_ids': x['iids'][:nr,:], 
                                         'labels': x['labels']})
    return dataset    

    
def split_dataset(dataset, size=None,
                  perc_train=.7, perc_val=.1, 
                  perc_test=.1, tuning=None):
    ''' Split dataset into training, validation and test set 
    Args:
        dataset (TFDataset): dataset to split (preprocessed and batched)
        size (int): number of examples from dataset. If None,
            the total number of examples is calculated and all examples 
            are used.
        perc_train (float): percentage of examples in training set
        perc_val (float): percentage of examples in training set
        perc_test (float): percentage of examples in test set
        tuning (optional): if provided, defines number of example for 
            additional tuning dataset.
    Returns:
        tuning, training, valdiation and test set
    ''' 
    if size is None:
        size = 0
        for _ in dataset:
            size += 1
        print(f'Number of total examples: {size}')
    size_train = int(size * perc_train)
    size_val = int(size * perc_val)
    size_test = int(size * perc_test)
    d_train = dataset.take(size_train)
    d_val = dataset.skip(size_train).take(size_val)
    d_test = dataset.skip(size_train + size_val).take(size_test)
    if tuning is None:
        return d_train, d_val, d_test
    else:
        d_tuning = dataset.take(tuning)
        return d_tuning, d_train, d_val, d_test
