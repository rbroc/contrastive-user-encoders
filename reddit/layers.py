import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import (Dense,
                                     Dropout,
                                     Lambda)
from reddit.utils import sampling_vae


class SimpleCompressor(layers.Layer):
    ''' Compresses encodings through relu layer(s)'''
    def __init__(self, compress_to, intermediate_size=None):
        dropout = Dropout(.20)
        compress = Dense(units=compress_to, activation='relu')
        if intermediate_size:
            intermediate = Dense(units=intermediate_size, 
                                 activation='relu')
            layers = [dropout, intermediate, compress]
        else:
            layers = [dropout, compress]
        self.compressor = tf.keras.models.Sequential(layers)
        super().__init__()

    def call(self, encodings):
        out = self.compressor(encodings)
        return out


class VAECompressor(layers.Layer):
    ''' Compresses encodings through VAE '''
    def __init__(self, 
                 compress_to, 
                 intermediate_size=None,
                 encoder_dim=768,
                 batch_size=1):
        self.compress_to = compress_to
        self.batch_size = batch_size
        if intermediate_size:
            self.encoder_int = Dense(intermediate_size, activation='relu')
        else:
            self.encoder_int = None
        self.z_mean = Dense(compress_to, activation='relu')
        self.z_log_sigma = Dense(compress_to, activation='relu')
        self.z = Lambda(sampling_vae)
        if intermediate_size:
            self.decoder_int = Dense(intermediate_size, activation='relu')
        else:
            self.decoder_int = None
        self.outlayer = Dense(encoder_dim, activation='relu')
        super().__init__()
    
    def call(self, encodings):
        if self.encoder_int:
            x = self.encoder_int(encodings)
        else:
            x = encodings
        zm = self.z_mean(x)
        zls = self.z_log_sigma(x)
        x = self.z([zm, zls, self.compress_to, self.batch_size])
        if self.decoder_int:
            x = self.decoder_int(x)
        out = self.outlayer(x)
        return out


