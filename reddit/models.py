import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense,
                                     Concatenate, 
                                     Lambda,
                                     LayerNormalization)
from reddit.utils import (average_encodings,
                          make_triplet_model_from_params)
from reddit.layers import (SimpleCompressor, 
                           VAECompressor)


class BatchTransformer(keras.Model):
    ''' Transformer model wrapping HuggingFace transformer to
        support 3D (batch, n_sentences, n_tokens) inputs.
        Args:
            transformer (model): model object from huggingface
                transformers (e.g. TFDistilBertModel)
            pretrained_weights (str): path to pretrained weights
            name (str): model name.
            trainable (bool): whether to freeze weights
            output_attentions (bool): if attentions should be added
                to outputs (useful for diagnosing but not much more)
            compress_to (int): dimensionality for compression
            compress_mode (str): if compress_to is defined, can be
                'dense' for linear compression or 'vae' for auto-encoder
                compression
            intermediate_size (int): size of intermediate layers for 
                compression
            pooling (str): can be cls, mean, random. Random just pulls 
                random non-zero tokens.
    '''
    def __init__(self, transformer, 
                 pretrained_weights,
                 trained_encoder_weights=None,
                 trained_encoder_class=None,
                 name=None, trainable=True,
                 output_attentions=False, 
                 compress_to=None,
                 compress_mode=None,
                 intermediate_size=None,
                 pooling='cls',
                 vocab_size=30522,
                 n_layers=None, 
                 batch_size=1):
        super(BatchTransformer, self).__init__(name=name)
        if name is None:
            cto_str = str(compress_to) + '_' if compress_to else 'no'
            cmode_str = compress_mode or ''
            int_str = str(intermediate_size) + '_' if intermediate_size else 'no'
            weights_str = pretrained_weights or trained_encoder_weights or 'scratch'
            weights_str = weights_str.replace('-', '_')
            layers_str = n_layers or 6
            name = f'BatchTransformer-{layers_str}layers-{cto_str}{cmode_str}dense'
            name = name + f'-{int_str}int-{pooling}-{weights_str}'
        self.encoder = make_triplet_model_from_params(transformer,
                                                      pretrained_weights,  
                                                      vocab_size, 
                                                      n_layers,
                                                      trained_encoder_weights,
                                                      trained_encoder_class,
                                                      output_attentions)
        self.trainable = trainable
        self.output_signature = tf.float32
        self.output_attentions = output_attentions
        if compress_to:
            if compress_mode == 'dense':
                self.compressor = SimpleCompressor(compress_to, 
                                                   intermediate_size)
            elif compress_mode == 'vae':
                self.compressor = VAECompressor(compress_to, 
                                                intermediate_size, 
                                                batch_size=batch_size)
        else:
            self.compressor = None
        if pooling == 'mean':
            self.layernorm = LayerNormalization(epsilon=1e-12)
        else:
            self.layernorm = None
        self.pooling = pooling

    def _encode_batch(self, example):
        mask = tf.reduce_all(tf.equal(example['input_ids'], 0), 
                             axis=-1, keepdims=True)
        mask = tf.cast(mask, tf.float32)
        mask = tf.abs(tf.subtract(mask, 1.))
        output = self.encoder(input_ids=example['input_ids'],
                              attention_mask=example['attention_mask'])
        if self.pooling == 'cls':
            encoding = output.last_hidden_state[:,0,:]
        elif self.pooling == 'mean':
            encoding = tf.reduce_sum(output.last_hidden_state[:,1:,:], axis=1)
            n_tokens = tf.reduce_sum(example['attention_mask'], axis=-1, keepdims=True)
            encoding = encoding / tf.cast(n_tokens, tf.float32)
        elif self.pooling == 'random':
            n_nonzero = tf.reduce_sum(example['attention_mask'], axis=-1, keepdims=True)
            fill_zero_mask = tf.cast(tf.multiply(tf.abs(tf.subtract(mask, 1.)), 2.), tf.int32)
            n_nonzero = tf.add(n_nonzero, fill_zero_mask) # 10 x 1
            idxs = tf.map_fn(lambda x: tf.random.uniform(shape=[], 
                                                         minval=1,
                                                         maxval=x[0],
                                                         dtype=tf.int32), 
                             n_nonzero)
            idxs = tf.one_hot(idxs, depth=512)
            idxs = tf.expand_dims(idxs, axis=-1)
            encoding = tf.multiply(output.last_hidden_state, idxs)
            encoding = tf.reduce_sum(encoding, axis=1)
            
        attentions = output.attentions if self.output_attentions else None
        masked_encoding = tf.multiply(encoding, mask)
        return masked_encoding, attentions

    
    def call(self, input):
        encodings, attentions = tf.vectorized_map(self._encode_batch, 
                                                  elems=input)
        if self.layernorm:
            encodings = self.layernorm(encodings)
        if self.compressor:
            encodings = self.compressor(encodings)
        if self.output_attentions:
            return encodings, attentions
        else:
            return encodings


class BatchTransformerClassifier(BatchTransformer):
    ''' Adds classification layer '''
    def __init__(self, 
                 nposts,
                 transformer, 
                 pretrained_weights,
                 trained_encoder_weights=None,
                 trained_encoder_class=None,
                 name=None, 
                 trainable=True,
                 compress_to=None,
                 compress_mode=None,
                 intermediate_size=None,
                 pooling='cls',
                 vocab_size=30522,
                 n_layers=None, 
                 batch_size=1,
                 use_embeddings='all'):
        name = name or f'BatchTransformerClassifier-{nposts}posts-{use_embeddings}emb'
        super().__init__(transformer, pretrained_weights,
                         trained_encoder_weights,
                         trained_encoder_class,
                         name, trainable, False,
                         compress_to, compress_mode, intermediate_size,
                         pooling, vocab_size, n_layers, batch_size)
        if use_embeddings == 'all':
            self.concat = Concatenate(axis=-1)
        else:
            self.concat = None
        self.dense = Dense(units=1, activation='sigmoid')
        self.use_embeddings = use_embeddings
        self.nposts = nposts
        
    def call(self, input):
        encodings = super().call(input)
        enc_1 = encodings[:, :self.nposts, :]
        enc_2 = encodings[:, self.nposts:, :]
        avg_enc_1 = tf.reduce_mean(enc_1, axis=1)
        avg_enc_2 = tf.reduce_mean(enc_2, axis=1)
        if self.use_embeddings == 'all':
            pre_logits = self.concat([avg_enc_1, avg_enc_2, tf.abs(avg_enc_1 - avg_enc_2)])
        elif self.use_embeddings == 'distance':
            pre_logits = tf.abs(avg_enc_1 - avg_enc_2)
        logits = self.dense(pre_logits)
        return logits
        

class BatchTransformerFFN(BatchTransformer):
    ''' Batch transformer with added dense layers
    Args:
        transformer (model): model object from huggingface
            transformers (e.g. TFDistilBertModel) for batch
            transformer
        pretrained_weights (str): path to pretrained weights
        n_dense (int): number of dense layers to add on top
            of batch transformer
        dims (int or list): number of nodes per layer
        activations (str, list or keras activation): type of 
            activation per layer
        trainable (bool): whether to freeze transformer weights
        name (str): model name. If not provided, concatenates
            path_to_weights, n_dense, dim, activation
        kwargs: kwargs for layers.Dense call
    '''
    def __init__(self,
                 transformer, 
                 pretrained_weights,
                 trained_encoder_weights=None,
                 trained_encoder_class=None,
                 n_dense=1,
                 dims=[768],
                 activations=['relu'],
                 trainable=False,
                 name=None,
                 n_layers=None,
                 vocab_size=30522):

        if len(dims) != n_dense:
            raise ValueError('length of dims does '
                                'match number of layers')
        if len(activations) != n_dense:
                raise ValueError('length of activations does '
                                 'match number of layers')           
        self.dims = dims
        self.activations = activations
        if name is None:
            weights_str = pretrained_weights or trained_encoder_weights or 'scratch'
            weights_str = weights_str.replace('-', '_')
            layers_str = n_layers or 6
            name = f'''BatchTransformerFFN-
                       {layers_str}layers-{n_dense}_
                       dim-{'_'.join([str(d) for d in dims])}_
                       {'_'.join(activations)}-{weights_str}'''
        super().__init__(transformer, pretrained_weights, 
                         trained_encoder_weights, trained_encoder_class,
                         name, vocab_size, n_layers, trainable)
        self.dense_layers = keras.Sequential([Dense(dims[i], activations[i])
                                              for i in range(n_dense)])
        self.average_layer = Lambda(average_encodings)
        self.concat_layer = Concatenate(axis=1)
        
    def call(self, input):
        encodings, _ = super().call(input)
        avg_anchor = self.average_layer(encodings)
        avg_pos = self.average_layer(encodings)
        avg_neg = self.average_layer(encodings)
        encodings = self.concat_layer([avg_neg, avg_pos, avg_anchor])
        encodings = self.dense_layers(encodings)
        return encodings
    
