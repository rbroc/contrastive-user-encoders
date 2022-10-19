import tensorflow as tf
from pathlib import Path
import json
import numpy as np
from glob import glob
import pickle as pkl
from abc import ABC, abstractmethod


class Logger:
    ''' Helper for loss/metric logging during training 
    Args:
        trainer (Trainer): instance of trainer class
    '''
    def __init__(self, trainer, path='..'):
        self.logdict = {}
        self.trainer = trainer
        self.outfolder = Path(path) / 'metrics' / self.trainer.model.name
        self.outfolder = self.outfolder / self.trainer.loss_object.name
        self.outfolder.mkdir(exist_ok=True, parents=True)
        self._reset()

    def _reset(self):
        ''' Initialize/empties metrics dictionary '''
        for v in self.trainer.train_vars + self.trainer.test_vars:
            self.logdict[v] = []
        for m in self.trainer.meta_vars:
            self.logdict[m] = []
        self.logdict['example_ids'] = []
        self.logdict['test_example_ids'] = []
    
    def log(self, logvars, epoch, example_ids, batch=None,
            train=True, meta=None):
        ''' Logs metrics dictionary 
        Args:
            logvars (list): metrics to log (list of lists of tensors)
            epoch (int): number of epoch
            batch (int): batch number (1 to n steps)
            example_id (int): example id
            train (bool): defines whether logging as 
                part of training. If so, saves file at
                specified intervals
            meta (list): meta variables to log.
         '''
        for idx, d in enumerate(logvars):
            try:
                fv = [float(v.numpy()) for v in d]
            except:
                fv = [v.numpy()[0].tolist() for v in d] # list vars
            if train:
                self.logdict[self.trainer.train_vars[idx]] += fv
            else:
                self.logdict[self.trainer.test_vars[idx]] += fv
        for idx, m in enumerate(meta):
            metaval = [i.numpy().tolist() for i in m]
            self.logdict[self.trainer.meta_vars[idx]] += metaval
        ids = [int(i.numpy()) for i in example_ids]
        if train:
            self.logdict['example_ids'] += ids
            if (batch == self.trainer.steps_per_epoch) or \
                (batch % self.trainer.log_every == 0):
                self._save(epoch)
        else:
            self.logdict['test_example_ids'] += ids

    def _save(self, epoch):
        ''' Saves dictionary 
        Args:
            epoch (int): epoch number 
        '''
        epoch_dir = self.outfolder / f'epoch-{epoch}'
        epoch_dir.mkdir(exist_ok=True, parents=True)
        outfile = epoch_dir / 'log.json'
        with open(outfile, 'w') as fh:
            fh.write(json.dumps(self.logdict))


class Checkpoint(ABC):
    ''' Metaclass for model and optimizer checkpoint helpers 
    Args:
        trainer (Trainer): trainer object 
        type (str): 'checkpoint' or 'optimizer' 
    '''
    def __init__(self, trainer, type, path='..'):
        self.trainer = trainer
        self.model_path = Path(self.trainer.model.name) / self.trainer.loss_object.name
        self.model_path = Path(path) / type / self.model_path
        self.model_path.mkdir(exist_ok=True, parents=True)
        super().__init__()

    @abstractmethod
    def _load(self):
        ''' Loads checkpoint '''
        pass
    
    @abstractmethod
    def save(self):
        ''' Saves checkpoint '''
        pass


class ModelCheckpoint(Checkpoint):
    ''' Checkpoint helper for model weights
    Args:
        trainer (Trainer): trainer object
        device (str): argument for experimental_io_device 
            in tf.train.CheckpointOptions call
    '''
    def __init__(self, trainer, device, path='..'):
        super().__init__(trainer, 'checkpoint', path)
        self.options = tf.train.CheckpointOptions(device)
        self.moptions = tf.saved_model.SaveOptions(experimental_io_device=device)
        if self.trainer.load_epoch is not None:
            self._load()

    def _load(self):
        ''' Loads model checkpoint '''
        epoch_pattern = f'epoch-{self.trainer.load_epoch}'
        ckpt_file = tf.train.latest_checkpoint(str(self.model_path/epoch_pattern))
        print(f'**** Loading {ckpt_file} ****')
        self.trainer.model.load_weights(ckpt_file, options=self.options).expect_partial()

    def save(self, epoch, batch):
        ''' Saves model checkpoint 
        Args:
            epoch (int): epoch number
            batch (int): batch number (1 to number of steps)
        ''' 
        epoch_dir = self.model_path / f'epoch-{epoch}'
        epoch_dir.mkdir(exist_ok=True, parents=True)
        file_pattern = f'batch-{batch}-of-{self.trainer.steps_per_epoch}'
        out_pattern = epoch_dir / file_pattern
        self.trainer.model.save_weights(filepath=out_pattern, options=self.options)
        try:
            self.trainer.model.encoder.save_pretrained(str(epoch_dir))
        except:
            pass
        

class OptimizerCheckpoint(Checkpoint):    
    ''' Checkpoint helper for optimizer
    Args:
        trainer (Trainer): trainer object
    '''
    def __init__(self, trainer, path='..'):
        super().__init__(trainer, 'optimizer', path)
        if self.trainer.load_epoch:
            self._load()

    def _load(self):
        ''' Loads optimizer checkpoint '''
        epoch_pattern = f'epoch-{self.trainer.load_epoch}'
        opt_files = glob(str(self.model_path / epoch_pattern / '*'))
        opt_idx = np.argmax([int(f.split('/')[-1].split('-')[1]) for f in opt_files])
        opt_file = opt_files[opt_idx]
        print(f'Loading {opt_file}')
        opt_weights = pkl.load(file=open(opt_file, 'rb'))
        self.trainer.optimizer._create_all_weights(self.trainer.model.trainable_variables)
        self.trainer.optimizer.set_weights(opt_weights)

    def save(self, epoch, batch):
        ''' Saves optimizer checkpoint 
        Args:
            epoch (int): epoch number
            batch (int): batch number (1 to n training steps)
        '''
        epoch_dir = self.model_path / f'epoch-{epoch}'
        epoch_dir.mkdir(exist_ok=True, parents=True)
        batch_file = f'batch-{batch}-of-{self.trainer.steps_per_epoch}.pkl'
        outfile = str(epoch_dir /  batch_file)
        with open(outfile, 'wb') as fh:
            pkl.dump(file=fh, obj=self.trainer.optimizer.get_weights())