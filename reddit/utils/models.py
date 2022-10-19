import tensorflow as tf
from pathlib import Path


def make_triplet_model_from_params(transformer,
                                   pretrained_weights,  
                                   vocab_size, 
                                   n_layers,
                                   trained_encoder_weights,
                                   trained_encoder_class,
                                   output_attentions):
    ''' Initializes a model given info on transformer class and whether to 
        import weights (if not, also uses info on vocab_size, n_layers etc)
    Args:
        transformer (transformers.Model): transformer class for mlm model
        pretrained_weights (str): path or id of pretrained to import
        vocab_size (int): vocab_size (if not pretrained)
        n_layers (int): n layers for encoder
        trained_encoder_weights (str): path to trained encoder weights
        trained_encoder_class (transformers.Model): class for encoder
    '''
    if pretrained_weights is None:
        config = transformer.config_class(vocab_size=vocab_size, 
                                          n_layers=n_layers)
        model = transformer(config, output_attentions=output_attentions)
    else:
        model = transformer.from_pretrained(pretrained_weights, output_attentions=output_attentions)
    if trained_encoder_weights and trained_encoder_class:
        load_weights_from_huggingface(model=model, 
                                      transformers_model_class=trained_encoder_class,
                                      weights_path=trained_encoder_weights)
    return model


def freeze_encoder_weights(encoder, freeze_param):
    ''' Freezes encoder layer, given an encoder and a list of 
        layers to freeze (no freezing if freeze_param is False or None) '''
    if not freeze_param:
        encoder.trainable = True
    else:
        for fl in freeze_param:
            encoder._layers[1]._layers[0][int(fl)]._trainable = False
        encoder._layers[0]._trainable = False # freeze embeddings


def dense_to_str(add_dense, dims):
    ''' Converts info on # dense layers to add and dimensions for 
        each to string (used for model ids)
    '''
    if not dims:
        dims_str = '0'
    else:
        assert len(dims) == add_dense
        dims_str = '_'.join([str(d) for d in dims])
    return dims_str


def save_encoder_huggingface(ckpt_path,
                             model=None,
                             reddit_model_class=None,
                             transformers_model_class=None, 
                             transformers_weights=None,
                             outpath=None):
    ''' Saves weights in format compatible with huggingface 
        transformers' from_pretrained method
    Args:
        ckpt_path: path to checkpoint
        model: initialized reddit model
        reddit_model_class: if model is not defined, pass Reddit model 
            class (e.g., BatchTransformer) here
        transformers_model_class: if model is not defined, pass huggingface's
            transformers class here (e.g., TFDistilBertModel)
        transformer_weights: if model is not defined,
            pass pretrained weights for transformers model here
    '''
    if outpath is None: 
        outpath = Path(ckpt_path) / '..' / '..'/ '..' / '..' / 'huggingface'
    outpath.mkdir(exist_ok=True, parents=True)
    if model is None:
        model = reddit_model_class(transformer=transformers_model_class,
                                   pretrained_weights=transformers_weights)
    ckpt = tf.train.latest_checkpoint(ckpt_path)
    model.load_weights(ckpt)
    model.encoder.save_pretrained(outpath) # only saves encoder weights, could be refined
    return 'Model saved!'


def load_weights_from_huggingface(model,
                                  transformers_model_class,
                                  weights_path,
                                  layer=0):
    ''' Load transformer weights from another huggingface model (for 
        entire model or one layer only). There may be more elegant 
        way to do this, but this creates a second model on the fly and
        gets the weights, then transfers them to the target model
    Args:
        model: transformer model (destination for weights import)
        transformers_model_class: transformer model to import weights from (source 
            for weights import)
        weights_path: path to read the weights from (or name of pretrained model)
        layer (optional): specifies which layer to import weights for, in case 
            only one layer's weights need to be loaded/updated
    '''
    if layer is not None:
        model.layers[layer]\
             .set_weights(transformers_model_class.from_pretrained(weights_path)\
                                                  .get_weights())
    else:
        model.set_weights(transformers_model_class.from_pretrained(weights_path)\
                                                  .get_weights())
