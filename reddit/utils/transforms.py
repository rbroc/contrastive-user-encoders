from reddit.utils import (pad_and_stack_triplet,
                          pad_subreddits,
                          pad_triplet_baselines, 
                          average_encodings_unbatched)


def triplet_transform(dataset, 
                      pad_to=[20,1,1],
                      batch_size=4, 
                      n_anchor=None,
                      filter_by_n=False):
    '''Transform pipeline for triplet dataset
    Args:
        dataset: dataset to transform
        pad_to (list): number of maximum posts to pad to
            for anchor, positive, negative example respectively 
        batch_size (int): global batch size (i.e., number of 
            replicas * effective batch size)
    '''
    dataset = pad_and_stack_triplet(dataset, 
                                    pad_to, 
                                    n_anchor,
                                    filter_by_n)
    return dataset.batch(batch_size, drop_remainder=True)


def subreddit_transform(dataset, pad_to=3, 
                        batch_size=4, nr=3):  
    ''' Transform pipeline for subreddit prediction'''
    dataset = pad_subreddits(dataset,
                             pad_to, 
                             nr)
    return dataset.batch(batch_size, drop_remainder=True)    


def triplet_baselines_transform(dataset, pad_to=3, 
                                batch_size=4, nr=3,
                                dedict=False, which='single'):  
    ''' Transform pipeline for subreddit prediction'''
    dataset = pad_triplet_baselines(dataset, pad_to, nr)
    if dedict:
        if which!='single':
            dataset = dataset.map(lambda x: (average_encodings_unbatched(x['input_ids']), 
                                             x['labels']))
        else:
            dataset = dataset.map(lambda x: (x['input_ids'][0], x['labels']))
    return dataset.batch(batch_size, drop_remainder=True)    

