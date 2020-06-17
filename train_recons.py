# Training procedure
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1' # MacOS pb
from dataset import BatchMaker
from models import *


def train_recons(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr, from_scratch=False):

  # Learning devices
  sched = tf.keras.experimental.CosineDecayRestarts(
    initial_learning_rate=init_lr, first_decay_steps=n_batches, t_mul=2.0, m_mul=0.95, alpha=0.3)
  optim = tf.keras.optimizers.Adam(sched)

  # To store the losses 
  losses = tf.Variable(tf.zeros((1000,1)))
  # Checkpoint (save and load model weights and losses)
  model_dir  = '%s/%s/ckpt_model' % (os.getcwd(), wrapp.model_name)
  ckpt_model = tf.train.Checkpoint(optimizer=optim, net=wrapp.model, loss=losses)
  mngr_model = tf.train.CheckpointManager(ckpt_model, directory=model_dir, max_to_keep=1)
  if not from_scratch:
    ckpt_model.restore(mngr_model.latest_checkpoint)
    if mngr_model.latest_checkpoint:
      print('\nModel %s restored from %s' % (wrapp.model_name, mngr_model.latest_checkpoint))
    else:
      print('\nModel %s initialized from scratch' % (wrapp.model_name))
      if not os.path.exists('./%s' % (wrapp.model_name,)):
        os.mkdir('./%s' % (wrapp.model_name,))
  else:
    print('\nModel %s initialized from scratch' % (wrapp.model_name))
    if not os.path.exists('./%s' % (wrapp.model_name,)):
      os.mkdir('./%s' % (wrapp.model_name,))

  # Training loop for the reconstruction part
  batch_maker = BatchMaker('recons', obj_type, n_objs, batch_size, wrapp.n_frames, im_dims)
  for e in range(n_epochs):
    # Mean batch loss
    mean_loss = 0.0
    loss_tens = np.zeros((1000,1))
    opt_e = optim.iterations/n_batches
    if opt_e % 10 == 0 and e > 0:
      mngr_model.save()
      print('\nCheckpoint saved at %s' % (mngr_model.latest_checkpoint,))
    for b in range(n_batches):  # batch shape: (batch_s, n_frames) + im_dims
      batch = tf.stack(batch_maker.generate_batch(), axis=1)/255
      rec_loss = wrapp.train_step(batch, b, optim)
      mean_loss += rec_loss
      if b == 0:
        lr_str = "{:.2e}".format(optim._decayed_lr(tf.float32).numpy())
        print('\nStarting epoch %03i, lr = %s, rec loss = %.3f' % (opt_e, lr_str, rec_loss))
      print('\r  Running batch %02i/%2i' % (b+1, n_batches), end='')
    loss_tens[e,:] = mean_loss/n_batches
    # Mean epoch loss
    losses.assign_add(tf.Variable(tf.cast(loss_tens, tf.float32)))

  # Plot the loss vs epochs
  wrapp.plot_results(tf.cast(optim.iterations, tf.float32).numpy()/n_batches, losses, 'loss', 'recons')


if __name__ == '__main__':

  obj_type    = 'neil'       # can be 'ball' or 'neil' for now
  n_objs      = 2            # number of moving object in each sample
  im_dims     = (64, 64, 1)  # image dimensions
  n_frames    = 5            # frames in the input sequences
  n_epochs    = 100          # epochs ran after latest checkpoint epoch
  batch_size  = 16           # sample sequences sent in parallel
  n_batches   = 64           # batches per epoch
  init_lr     = 2e-4         # first parameter to tune if does not work
  model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet'
  decoder     = conv_decoder()
  wrapp       = Wrapper(model, my_recons, decoder, n_frames, name)
  train_recons(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr, from_scratch=True)
