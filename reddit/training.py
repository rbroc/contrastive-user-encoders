import tensorflow as tf
from tensorflow.keras.utils import Progbar
from reddit import (Logger, ModelCheckpoint,
                    OptimizerCheckpoint)
from reddit.utils import LOG_DICT, META_DICT, PBAR_DICT
                    

class Trainer:
    ''' Trainer class (can be run in predict only mode)
    Args:
        model: model object
        loss_object: loss object
        optimizer: optimizer object
        strategy (tf.Strategy): distribution strategy. Can be 
            set to None in non-distributed contexts
        n_epochs (int): number of training epochs
        steps_per_epoch (int): number of training steps/batches
        distributed (bool): whether training in distributed 
            strategy
        checkpoint_every (int): how often (in examples) model 
            and optimizer weights should be saved
        log_every (int): how often (in examples) training/test
            variables (loss, metrics, etc.) should be logged
        start_epoch (int): which epoch to start training from
        ds_type (str): type of dataset (triplet)
            This is used to determine which variables are logged 
            and displayed in the progress bar.
        checkpoint_device (str): argument to CheckpointOptions
        log_path (str): argument to Path, defines path where 
            metrics and checkpoints folder for logging are 
            located (created if not existing)
        eval_before_training (bool): whether to run a test epoch 
            before the first training epoch (useful to gather 
            baseline performance)
        update_every (int): parameter for frequency of gradient 
            accumulation
    '''
    def __init__(self, model, 
                 loss_object, strategy, 
                 steps_per_epoch,
                 n_epochs=1,
                 optimizer=None,
                 test_steps=None,
                 distributed=True,
                 checkpoint_every=None, 
                 log_every=100,
                 start_epoch=0,
                 ds_type='triplet',
                 checkpoint_device=None,
                 log_path='..',
                 eval_before_training=True,
                 update_every=1):
        self.train_vars = LOG_DICT[ds_type]
        self.test_vars = ['test_' + v for v in self.train_vars]
        self.meta_vars = META_DICT[ds_type]
        self.pbar_vars = PBAR_DICT[ds_type]
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.strategy = strategy
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.test_steps = test_steps
        self.distributed = distributed
        self.checkpoint_every = checkpoint_every or steps_per_epoch
        self.log_every = log_every
        self.start_epoch = start_epoch
        if start_epoch > 0:
            self.load_epoch = start_epoch - 1
        else:
            self.load_epoch = None    
        self.logger = Logger(self, log_path)
        self.model_ckpt = ModelCheckpoint(self, checkpoint_device, log_path)
        self.opt_ckpt = OptimizerCheckpoint(self, log_path)
        self.eval_before_training = eval_before_training
        self.update_every = update_every


    def _train_step(self, batch_in_replica, labels):
        ''' Define training step (single replica) '''
        with tf.GradientTape() as tape:
            model_out = self.model(batch_in_replica)
            if labels:
                loss_out = self.loss_object(model_out, 
                                            batch_in_replica['labels'])
            else:
                loss_out = self.loss_object(model_out)
        gradients = tape.gradient(loss_out[0], 
                                  self.model.trainable_variables)
        return loss_out, gradients


    def _test_step(self, batch_in_replica, labels):
        ''' Define test step (single replica) '''
        model_out = self.model(batch_in_replica)
        if labels:
            test_loss_out = self.loss_object(model_out, 
                                             batch_in_replica['labels'])
        else:
            test_loss_out = self.loss_object(model_out) 
        return test_loss_out


    @tf.function
    def _run_distributed_train_step(self, global_batch, labels):
        ''' Run training/test step on all replicas '''
        step_outs, gradients = self.strategy.run(self._train_step, 
                                                 args=(global_batch,
                                                       labels))
        gradsum = [self.strategy.reduce(tf.distribute.ReduceOp.MEAN, 
                                        g,
                                        axis=None) for g in gradients] # averaging within batch
        return [getattr(o,'values') for o in step_outs], gradsum

    
    @tf.function
    def _run_distributed_test_step(self, global_batch, labels):
        ''' Run test step on all replicas '''
        step_outs = self.strategy.run(self._test_step, args=(global_batch,
                                                             labels))
        return [getattr(o,'values') for o in step_outs]

    def _gradient_update(self, accumulated_grads):
        ''' Define gradient update '''
        if self.update_every != 1:
            avg_grads = [a/(self.update_every) for a in accumulated_grads]
        else:
            avg_grads = [a for a in accumulated_grads] # division crashes at first step
        self.optimizer.apply_gradients(zip(avg_grads, 
                                           self.model.trainable_variables))
        
    @tf.function
    def _run_distributed_gradient_update(self, accumulated_grads):
        ''' Run gradient update on all replicas'''
        self.strategy.run(self._gradient_update, args=(accumulated_grads,))
        
    
    def _accumulate_gradients(self, gradients, accumulated_grads):
        ''' Update gradients by summing new gradient '''
        return [tf.add(*g) for g in zip(gradients, 
                                        accumulated_grads)]
    
    def _reset_gradients(self, accumulated_grads):
        ''' Set accumulated gradients to 0'''
        return [tf.multiply(w,0.) for w in accumulated_grads]
        
        
    def _run_train_epoch(self, epoch, dataset_train, labels):
        ''' Run one training epoch 
        Args:
            epoch (int): epoch number
            dataset_train (DistributedDataset): training set
            labels (bool): whether the example includes labels
        '''
        pb = Progbar(self.steps_per_epoch, 
                     stateful_metrics=self.pbar_vars)
        
        for n, example in enumerate(dataset_train):
            if self.distributed:
                outs, grads = self._run_distributed_train_step(example, labels)
                if n == 0:
                    accumulated_grads = grads # initialize gradients 
                else:
                    accumulated_grads = self._accumulate_gradients(grads, 
                                                                   accumulated_grads)
                ids = list(tf.concat(example['id'].values, 
                                     axis=0)) if 'id' in example.keys() else []
                meta = [list(tf.concat(example[mvar].values, axis=0)) 
                        for mvar in self.meta_vars]
            else:
                outs, grads = [[o] for o in self._train_step(example, labels)]
                if n == 0:
                    accumulated_grads = grads
                else:
                    accumulated_grads = self._accumulate_gradients(grads, 
                                                                   accumulated_grads)
                ids = list(example['id']) if 'id' in example.keys() else []
                meta = [list(example[mvar]) for mvar in self.meta_vars]
            
            if ((n+1) % self.update_every) == 0:
                self._run_distributed_gradient_update(accumulated_grads)
                accumulated_grads = self._reset_gradients(accumulated_grads)
            
            self.logger.log(list(outs), epoch, ids, n+1, meta=meta)
            pbar_values = [tf.reduce_mean(o).numpy() for o in outs]
            pb.add(1, values=list(zip(self.pbar_vars, pbar_values)))
            
            if ((n+1) % self.checkpoint_every == 0) or \
                (n+1 == self.steps_per_epoch):
                self.model_ckpt.save(epoch, n+1)
                self.opt_ckpt.save(epoch, n+1)
        
        self.model_ckpt.save(epoch, n+1)
        print('; '.join([f'''Mean {m}: {tf.reduce_mean(self.logger.logdict[m]).numpy()}'''
                          for m in self.pbar_vars]))
        

    def _run_test_epoch(self, epoch, dataset_test, labels):
        ''' Run one validation/test epoch 
        Args:
            epoch (int): epoch number 
            dataset_test (DistributedDataset): test set 
            labels (bool): whether the example includes labels
        '''
        if self.test_steps:
            pb = Progbar(self.test_steps)
            
        for example in dataset_test:
            if self.distributed:
                outs = self._run_distributed_test_step(example, labels)
                ids = list(tf.concat(example['id'].values, 
                                     axis=0)) if 'id' in example.keys() else []
                meta = [list(tf.concat(example[mvar].values, axis=0)) 
                        for mvar in self.meta_vars]
            else:
                outs = [[o] for o in self._test_step(example, labels)]
                ids = list(example['id']) if 'id' in example.keys() else []
                meta = [list(example[mvar]) for mvar in self.meta_vars]
            self.logger.log(list(outs), epoch, ids, train=False, meta=meta)
            if self.test_steps:
                pb.add(1)

        self.logger._save(epoch)
        print('; '.join([f'''Mean test {m}: {tf.reduce_mean(self.logger.logdict[f'test_{m}']).numpy()}'''
                          for m in self.pbar_vars]))

    def run(self, 
            dataset_train=None, 
            dataset_test=None, 
            shuffle=True,
            transform=None,
            transform_test=False,
            test_only=False, 
            test_epoch_name='test_only',
            labels=False,
            **transform_kwargs):
        ''' Run full training 
        Args:
            dataset_train (Dataset): training set (not distributed)
            dataset_test (Dataset): validation set (not distributed)
            shuffle (bool): whether to shuffle the dataset at each epoch
            transform (function): if defined, the function passed
                here is applied to the training dataset.
            transform_test (bool): apply transformation to test too?
            test_only (bool): whether to only run test epoch
            test_epoch_name (str): identifier for test epoch
            labels (bool): whether the training is supervised (and labels are 
                thus included in the input dictionary for each example)
            transform_kwargs: keyword arguments for transform function call
        '''
        
        if dataset_test:
            if transform and transform_test:
                dataset_test = transform(dataset_test,
                                         **transform_kwargs)
            if self.distributed:
                dataset_test = self.strategy.experimental_distribute_dataset(dataset_test)
        
        if test_only is False:
            if self.eval_before_training:
                self._run_test_epoch('baseline', dataset_test, labels)
                self.logger._reset()
            
            for epoch in range(self.start_epoch, self.n_epochs):
                dataset = dataset_train
                if transform:
                    dataset = transform(dataset,
                                        **transform_kwargs)
                if shuffle:
                    print('Shuffling training data...')
                    dataset = dataset.shuffle(int(self.steps_per_epoch/2)) # int(1)
                    
                if self.distributed:
                    dataset = self.strategy.experimental_distribute_dataset(dataset)
                    
                print(f'Epoch {epoch+1}/{self.n_epochs}')
                self._run_train_epoch(epoch, dataset, labels)
                
                if dataset_test:
                    self._run_test_epoch(epoch, dataset_test, labels)
                self.logger._reset()
                
        else:
            self._run_test_epoch(test_epoch_name, dataset_test, labels)
            self.logger._reset()
