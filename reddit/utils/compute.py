import tensorflow as tf


def sampling_vae(args):
    ''' Sampling function for VAE'''
    z_mean, z_log_sigma, compress_to, bs = args
    epsilon = tf.random.normal(shape=(bs,
                                      compress_to),
                               mean=0., stddev=0.1)
    return z_mean + tf.math.exp(z_log_sigma) * epsilon


def average_encodings(encodings):
    ''' Averages encodings along feature dimension
    Args:
        encodings (tf.Tensor): tensor of anchor encodings
            (of shape bs x n_posts x n_dims)
    '''
    out = tf.reduce_sum(encodings, axis=1, keepdims=True)
    mask = tf.reduce_all(tf.equal(encodings, 0), axis=-1, 
                         keepdims=True)
    mask = tf.cast(mask, tf.float32)
    mask = tf.abs(tf.subtract(mask, 1))
    n_nonzero = tf.reduce_sum(mask, axis=1, keepdims=True)
    out = tf.divide(out, n_nonzero)
    return out


def compute_mean_pairwise_distance(encodings):   
    ''' Computes mean distance between embeddings 
    Args:
        encodings (tf.Tensor): tensor of encodings 
            (n_posts x n_dims)
    '''     
    sqr_enc = tf.reduce_sum(encodings*encodings, axis=1)
    mask = tf.cast(tf.not_equal(sqr_enc, 0), tf.float32)
    sqr_enc = tf.reshape(sqr_enc, [-1,1])
    dists = sqr_enc - 2*tf.matmul(*[encodings]*2, transpose_b=True)
    dists = dists + tf.transpose(sqr_enc)
    dists = tf.transpose(dists * mask) * mask
    dists = tf.linalg.band_part(dists,-1,0)
    dists = dists - tf.linalg.band_part(dists,0,0)
    range_valid = tf.range(mask.shape[-1], dtype=tf.float32) * mask
    n_valid_dists = tf.reduce_sum(range_valid)
    mean_dist = tf.divide(tf.reduce_sum(dists), n_valid_dists)
    return mean_dist


def average_encodings_unbatched(encodings):
    ''' Averages encodings along feature dimension
    Args:
        encodings (tf.Tensor): tensor of anchor encodings
            (of shape n_posts x n_dims)
    '''
    out = tf.cast(tf.reduce_sum(encodings, axis=0), tf.float32)
    mask = tf.reduce_all(tf.equal(encodings, 0), axis=-1)
    mask = tf.cast(mask, tf.float32)
    mask = tf.abs(tf.subtract(mask, 1))
    n_nonzero = tf.reduce_sum(mask)
    out = tf.divide(out, n_nonzero)
    return out
