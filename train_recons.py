# Training procedure
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1'  # MacOS pb
from dataset import BatchMaker
from models import *


def train_recons(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr, from_scratch=False):

  # Learning devices
  sched = tf.keras.experimental.CosineDecayRestarts(
    initial_learning_rate=init_lr, first_decay_steps=n_batches, t_mul=2.0, m_mul=0.95, alpha=0.3)
  optim = tf.keras.optimizers.Adam(sched)
  
  # Initialize checkpoint
  model_dir  = '%s/%s/ckpt_model' % (os.getcwd(), wrapp.model_name)
  ckpt_model = tf.train.Checkpoint(optim=optim, net=wrapp.model, losses=tf.Variable(tf.zeros((1000,))),)
  mngr_model = tf.train.CheckpointManager(ckpt_model, directory=model_dir, max_to_keep=1)
  
  # Try to load latest checkpoint (if required)
  if not from_scratch:
    ckpt_model.restore(mngr_model.latest_checkpoint)
    if mngr_model.latest_checkpoint:
      print('\nModel %s restored from %s\n' % (wrapp.model_name, mngr_model.latest_checkpoint))
    else:
      print('\nModel %s initialized from scratch\n' % (wrapp.model_name))
      if not os.path.exists('./%s' % (wrapp.model_name,)):
        os.mkdir('./%s' % (wrapp.model_name,))
  else:
    print('\nModel %s initialized from scratch\n' % (wrapp.model_name))
    if not os.path.exists('./%s' % (wrapp.model_name,)):
      os.mkdir('./%s' % (wrapp.model_name,))
  
  # Training loop for the reconstruction part
  batch_maker = BatchMaker('recons', obj_type, n_objs, batch_size, wrapp.n_frames, im_dims)
  for _ in range(n_epochs):
    e = ckpt_model.optim.iterations//n_batches

    # Train the model for one epoch
    mean_loss = 0.0
    for b in range(n_batches):  # batch shape: (batch_s, n_frames) + im_dims
      batch      = tf.stack(batch_maker.generate_batch(), axis=1)/255
      rec_loss   = wrapp.train_step(batch, b, ckpt_model.optim)
      mean_loss += rec_loss
      print('\rRunning batch %02i/%02i' % (b+1, n_batches), end='')

    # Record loss for this epoch
    mean_loss = mean_loss/n_batches
    lr_str    = "{:.2e}".format(ckpt_model.optim._decayed_lr(tf.float32).numpy())
    print('\n  Finishing epoch %03i, lr = %s, loss = %.3f' % (e, lr_str, mean_loss))
    loss_tens = tf.concat([tf.zeros((e,)), mean_loss*tf.ones((1,)), tf.zeros((1000-e-1,))], axis=0)
    ckpt_model.losses.assign_add(tf.Variable(loss_tens))

    # Save checkpoint if necessary
    if e % 10 == 9:
      mngr_model.save()
      print('\nModel checkpoint saved at %s' % (mngr_model.latest_checkpoint,))

  # Plot the loss curve
  wrapp.plot_results(range(e+1), ckpt_model.losses.numpy()[:e+1], 'epoch', 'loss', 'recons')


if __name__ == '__main__':

  obj_type     = 'neil'       # can be 'ball' or 'neil' for now
  n_objs       = 2            # number of moving object in each sample
  im_dims      = (64, 64, 1)  # image dimensions
  n_frames     = 3           # frames in the input sequences
  n_epochs     = 20          # epochs ran IN ADDITION TO latest checkpoint epoch
  batch_size   = 16           # sample sequences sent in parallel
  n_batches    = 64           # batches per epoch
  init_lr      = 2e-4         # first parameter to tune if does not work
  model, name  = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet2'
  wrapp        = Wrapper(model, my_recons, None, n_frames, name)
  train_recons(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr, from_scratch=False)
