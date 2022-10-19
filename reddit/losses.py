import tensorflow as tf
from reddit.utils import (average_encodings, 
                          compute_mean_pairwise_distance)
from abc import ABC, abstractmethod


class TripletLoss(ABC):
    ''' Base class for triplet loss 
    Args:
        margin (float): margin to be induced between distances of
            positive and negative encoding from avg of anchor encodings
        custom_loss_fn (function): custom loss function
    '''
    def __init__(self, margin,
                 custom_loss_fn=None, name=None):
        self.name = name or f'triplet_loss_margin-{margin}'
        self.margin = margin
        self.custom_loss_fn = custom_loss_fn
        super().__init__()

    
    def _loss_function(self, dist_pos, dist_neg, dist_anch=None):
        ''' Defines how loss is computed from encoding distances '''
        if self.custom_loss_fn:
            return self.custom_loss_fn(dist_pos, dist_neg, dist_anch)
        else:
            return tf.maximum(0.0, self.margin + (dist_pos-dist_neg))
    
    @abstractmethod
    def __call__(self):
        ''' Computes loss '''
        pass
    

class TripletLossBase(TripletLoss):
    ''' Triplet loss for BatchTransformer with no head
    Args:
        margin (float): margin to be induced between distances of
            positive and negative encoding from avg of anchor encodings
    '''
    def __init__(self, margin, n_neg=1, n_pos=1, n_anc=None,
                 custom_loss_fn=None, name=None):
        super().__init__(margin, custom_loss_fn, name)
        self.n_neg = n_neg
        self.n_pos = n_pos
        self.n_anc = n_anc
    
    def __call__(self, encodings):
        ''' Computes loss. Returns loss, metric and encodings distances 
        Args:
            encodings (tf.Tensor): posts encodings
        '''       
        neg_idx = self.n_neg
        pos_idx = neg_idx + self.n_pos
        n_enc = encodings[:, :neg_idx, :]
        p_enc = encodings[:, neg_idx:pos_idx, :]
        if self.n_anc:
            a_enc = encodings[:, pos_idx:pos_idx+self.n_anc, :]
        else:
            a_enc = encodings[:, pos_idx:, :]
        avg_a_enc = tf.squeeze(average_encodings(a_enc), axis=1)
        avg_n_enc = tf.squeeze(average_encodings(n_enc), axis=1)
        avg_p_enc = tf.squeeze(average_encodings(p_enc), axis=1)
        dist_pos = tf.reduce_sum(tf.square(avg_a_enc - avg_p_enc), axis=1)
        dist_neg = tf.reduce_sum(tf.square(avg_a_enc - avg_n_enc), axis=1)
        if (self.n_anc is None) or (self.n_anc > 1):
            dist_anch = tf.vectorized_map(compute_mean_pairwise_distance, 
                                          elems=a_enc)
        else:
            dist_anch = tf.zeros(shape=dist_pos.shape)
        metric = tf.cast(tf.greater(dist_neg, dist_pos), tf.float32)
        loss = self._loss_function(dist_pos, dist_neg, dist_anch)
        outs = [tf.reduce_mean(o, axis=0) 
                for o in [loss, metric, dist_pos, dist_neg, dist_anch]]
        return outs
    

class TripletLossFFN(TripletLoss):
    ''' Triplet loss for batch transformer with FFN head
    Args:
        margin (float): margin to be induced between distances of
            positive and negative encoding from avg of anchor encodings
    '''    
    def __init__(self, margin, custom_loss_fn=None, name=None):
        super().__init__(margin, custom_loss_fn, name)

    def __call__(self, encodings):
        ''' Computes loss. Returns loss, metric and encodings distances
        Args:
            encodings (tf.Tensor): posts encodings
        '''
        n_enc, p_enc, a_enc = encodings[:,0,:], encodings[:,1,:], encodings[:,2,:]
        dist_pos = tf.reduce_sum(tf.square(a_enc - p_enc), axis=1)
        dist_neg = tf.reduce_sum(tf.square(a_enc - n_enc), axis=1)
        metric = tf.cast(tf.greater(dist_neg, dist_pos), tf.float32)
        loss = self._loss_function(dist_pos, dist_neg)
        outs = [tf.reduce_mean(o, axis=0) 
                for o in [loss, metric, dist_pos, dist_neg]]
        return outs

    
class SubredditClassificationLoss:
    ''' Classification loss for subreddit classificationk '''
    def __init__(self, name=None, n_labels=20, pos_weights=None):
        super().__init__()
        self.name = name or 'classification-loss'
        self.n_labels = n_labels
        self.pos_weights = pos_weights
    
    def __call__(self, probs, labels):
        loss = tf.nn.weighted_cross_entropy_with_logits(labels, 
                                                        probs,
                                                        pos_weight=self.pos_weights)
        is_one = tf.cast(tf.greater(probs, .0), tf.float32)
        tp = tf.reduce_sum(labels * is_one, axis=-1) / self.n_labels # correct
        fn = tf.reduce_sum(tf.cast(((is_one - labels) == -1), 
                                   tf.float32), axis=-1) / self.n_labels
        fp = tf.reduce_sum(tf.cast(((is_one - labels) == 1), 
                                   tf.float32), axis=-1) / self.n_labels
        tn = 1 - tp - fn - fp

        metric = tf.cast(is_one == tf.cast(labels, tf.float32),
                         tf.float32)
        precision = tf.divide(tf.reduce_sum(labels * is_one, axis=-1),
                              tf.reduce_sum(is_one, axis=-1))
        recall = tf.divide(tf.reduce_sum(labels * is_one, axis=-1),
                           tf.reduce_sum(labels, axis=-1))
        outs = [tf.reduce_mean(o, axis=0) 
                for o in [loss, recall, precision, metric, tp, tn, fp, fn]]
        return outs

