LOG_DICT = {'triplet': ['losses', 
                        'metrics', 
                        'dist_pos', 
                        'dist_neg', 
                        'dist_anch'],
           'subreddit_classification': ['losses', 'recall', 'precision', 'accuracy', 
                                        'tp', 'tn', 'fp', 'fn']}

META_DICT = {'triplet': [],
             'subreddit_classification': []}

PBAR_DICT = {'triplet': ['losses', 'metrics'],
             'subreddit_classification': ['losses', 'recall', 'precision', 'accuracy', 
                                          'tp', 'tn', 'fp', 'fn']}