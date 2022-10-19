from reddit.utils import (load_tfrecord,
                          triplet_transform, 
                          filter_triplet_by_n_anchors)
from reddit.models import (BatchTransformer, BatchTransformerFFN)
from reddit.losses import (TripletLossBase, TripletLossFFN)
from reddit.training import Trainer
from transformers import TFDistilBertModel
import glob
from pathlib import Path
import argparse
import tensorflow as tf
from official.nlp.optimization import create_optimizer

DATA_PATH = Path('..') /'reddit'/ 'data' / 'datasets'/ 'triplet'

# Initialize parser
parser = argparse.ArgumentParser()

# Training loop argument
parser.add_argument('--dataset-name', type=str, default=None,
                    help='Name of dataset to use')
parser.add_argument('--triplet-type', type=str, default='standard',
                    help='Should be standard or FFN')
parser.add_argument('--log-path', type=str, default=None,
                    help='Path for metrics and checkpoints within ../logs')
parser.add_argument('--per-replica-batch-size', type=int, default=20,
                    help='Batch size')
parser.add_argument('--n-epochs', type=int, default=3,
                    help='Number of epochs')        
parser.add_argument('--start-epoch', type=int, default=0,
                    help='Epoch to start from')
parser.add_argument('--update-every', type=int, default=16,
                    help='Update every n steps')
# Loss arguments
parser.add_argument('--loss-margin', type=float, default=1.0,
                    help='Margin for triplet loss')  
parser.add_argument('--pad-anchor', type=int, default=10,
                    help='Max number of anchor posts')
parser.add_argument('--n-anchor', type=int, default=None,
                    help='Number of anchor posts used in loss')
parser.add_argument('--n-pos', type=int, default=1,
                    help='Number of positive examples')
parser.add_argument('--n-neg', type=int, default=1,
                    help='Number of negative examples')
# Model arguments
parser.add_argument('--pretrained-weights', type=str, 
                    default=None,
                    help='Pretrained huggingface model')
parser.add_argument('--trained-encoder-weights', type=str, default=None,
                    help='Path to trained encoder weights to load (hf format)')
parser.add_argument('--compress-to', type=int, default=None,
                    help='Dimensionality of compression head')
parser.add_argument('--compress-mode', type=str, default=None,
                    help='Whether to compress with dense or vae')
parser.add_argument('--which-set', type=str, default='val',
                    help='Which set to validate on')
parser.add_argument('--intermediate-size', type=int, default=None,
                    help='Dimensionality of intermediate layer in head')
parser.add_argument('--pooling', type=str, default='cls',
                    help='Whether to compress via pooling or other ways')
parser.add_argument('--vocab-size', type=int, default=30522,
                    help='Vocab size (relevant if new architecture')
parser.add_argument('--n-layers', type=int, default=None,
                    help='Nr layers if not pretrained')
# Arguments for FFN triplet
parser.add_argument('--n-dense', type=int, default=None,
                    help='''Number of dense layers to add,
                            relevant for FFN''')
parser.add_argument('--dims', nargs='+', help='Number of nodes in layers', 
                    default=None)
parser.add_argument('--activations', nargs='+', help='Activations in layers', 
                    default=None)
# Define boolean args
parser.add_argument('--test-only', dest='test_only', action='store_true',
                    help='Whether to only run one test epoch')
parser.add_argument('--filter-by-n', dest='filter_by_n', action='store_true',
                    help='Whether to select only posts that have n_anchor')
parser.set_defaults(test_only=False)
parser.set_defaults(filter_by_n=False)


def _run_training(log_path, 
                  dataset_name,
                  triplet_type,
                  per_replica_batch_size, 
                  n_epochs,
                  start_epoch,
                  pad_anchor,
                  n_anchor,
                  filter_by_n,
                  n_pos,
                  n_neg,
                  loss_margin,
                  pretrained_weights,
                  trained_encoder_weights,
                  compress_to,
                  compress_mode,
                  intermediate_size,
                  pooling,
                  n_dense,
                  dims,
                  activations,
                  update_every,
                  test_only,
                  vocab_size, 
                  n_layers,
                  which_set):
    
    # Define type of training
    if triplet_type == 'standard':
        model_class = BatchTransformer
        loss = TripletLossBase(margin=loss_margin, 
                               n_pos=n_pos, 
                               n_neg=n_neg,
                               n_anc=n_anchor)
    elif triplet_type == 'ffn':
        model_class = BatchTransformerFFN
        loss = TripletLossFFN(margin=loss_margin)
   
    # Config
    METRICS_PATH = Path('..') / 'logs' / 'triplet' / log_path / triplet_type
    METRICS_PATH.mkdir(parents=True, exist_ok=True)
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpus))
    try:
        tf.config.experimental.set_visible_devices(gpus, 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        print(e)
    strategy = tf.distribute.MirroredStrategy(devices=logical_gpus)
    
    # Set up dataset 
    pattern = str(DATA_PATH / dataset_name / 'train'/ 'batch*')
    fs_train = glob.glob(pattern)
    fs_val = glob.glob(str(DATA_PATH / dataset_name / which_set / 'batch*'))
    ds_train = load_tfrecord(fs_train, deterministic=True, ds_type='triplet')
    ds_val = load_tfrecord(fs_val, deterministic=True, ds_type='triplet')
    
    # Compute number of batches
    global_batch_size = len(logical_gpus) * per_replica_batch_size
    if n_anchor and filter_by_n:
        n_train_examples = len([e for e in ds_train 
                                if filter_triplet_by_n_anchors(e, n_anchor)])
        n_test_examples = len([e for e in ds_val 
                               if filter_triplet_by_n_anchors(e, n_anchor)])
    else:
        n_train_examples = len([e for e in ds_train])
        n_test_examples = len([e for e in ds_val])
    print(f'{n_train_examples}, n test examples: {n_test_examples}')
    n_train_steps = int(n_train_examples / global_batch_size)
    n_test_steps = int(n_test_examples / global_batch_size)
    
    # initialize optimizer, model and loss object
    with strategy.scope():
        optimizer = create_optimizer(2e-5, # allow params edit
                                     num_train_steps=n_train_steps * n_epochs,
                                     num_warmup_steps=10000) # could make it dependent on length of edit
        
        if triplet_type == 'standard':
            model = model_class(transformer=TFDistilBertModel,
                                pretrained_weights=pretrained_weights,
                                trained_encoder_weights=trained_encoder_weights,
                                trained_encoder_class=TFDistilBertModel,
                                trainable=True,
                                output_attentions=False,
                                compress_to=compress_to,
                                compress_mode=compress_mode,
                                intermediate_size=intermediate_size,
                                pooling=pooling,
                                vocab_size=vocab_size,
                                n_layers=n_layers)
        elif triplet_type == 'ffn':
            model = model_class(transformer=TFDistilBertModel,
                                pretrained_weights=pretrained_weights,
                                trained_encoder_weights=trained_encoder_weights,
                                trained_encoder_class=TFDistilBertModel,
                                n_dense=n_dense,
                                dims=dims,
                                activations=activations,
                                trainable=False,
                                vocab_size=vocab_size,
                                n_layers=n_layers)

    # Initialize trainer
    trainer = Trainer(model=model,
                      loss_object=loss,
                      optimizer=optimizer,
                      strategy=strategy, 
                      n_epochs=n_epochs, 
                      start_epoch=start_epoch,
                      steps_per_epoch=n_train_steps, 
                      log_every=1000,
                      ds_type='triplet',
                      log_path=str(METRICS_PATH),
                      checkpoint_device=None,
                      distributed=True,
                      eval_before_training=False,
                      test_steps=n_test_steps,
                      update_every=update_every)
    
    # Run training
    trainer.run(dataset_train=ds_train, 
                dataset_test=ds_val,
                shuffle=False,
                transform=triplet_transform,
                transform_test=True,
                test_only=test_only,
                labels=False, 
                pad_to=[pad_anchor, 
                        n_pos, 
                        n_neg],
                batch_size=global_batch_size,
                n_anchor=n_anchor,
                filter_by_n=filter_by_n)
    

if __name__=='__main__':
    args = parser.parse_args()
    _run_training(args.log_path, 
                  args.dataset_name,
                  args.triplet_type,
                  args.per_replica_batch_size, 
                  args.n_epochs,
                  args.start_epoch,
                  args.pad_anchor,
                  args.n_anchor,
                  args.filter_by_n,
                  args.n_pos,
                  args.n_neg,
                  args.loss_margin,
                  args.pretrained_weights,
                  args.trained_encoder_weights,
                  args.compress_to,
                  args.compress_mode,
                  args.intermediate_size,
                  args.pooling,
                  args.n_dense,
                  args.dims,
                  args.activations,
                  args.update_every,
                  args.test_only,
                  args.vocab_size,
                  args.n_layers,
                  args.which_set)
