# Training procedure
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1' # MacOS pb
from dataset import BatchMaker
from models import *


def train_decode(wrapp, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr, decode_mode, from_scratch=False):

  # Learning devices
  sched = CustomSchedule(init_lr, n_epochs, n_batches)
  optim = tf.keras.optimizers.Adam(sched)

  # Checkpoint (save and load model weights and accuracies)
  model_dir    = '%s/%s/ckpt_model' % (os.getcwd(), wrapp.model_name)
  decoder_dir  = '%s/%s/ckpt_decod' % (os.getcwd(), wrapp.model_name)
  ckpt_model   = tf.train.Checkpoint(net=wrapp.model)
  ckpt_decoder = tf.train.Checkpoint(optim=optim, net=wrapp.decoder, losses=tf.Variable(tf.zeros((1000,))), accs=tf.Variable(tf.zeros((1000,))), )
  mngr_model   = tf.train.CheckpointManager(ckpt_model,   directory=model_dir,   max_to_keep=1)
  mngr_decoder = tf.train.CheckpointManager(ckpt_decoder, directory=decoder_dir, max_to_keep=1)

  # Try to load latest checkpoints (if required)
  ckpt_model.restore(mngr_model.latest_checkpoint).expect_partial()
  if mngr_model.latest_checkpoint:
    print('\nReconstruction model loaded from %s\n' % (mngr_model.latest_checkpoint))
  else:
    print('\nWarning: your reconstruction model is not trained yet. Loading from scratch\n')
  if not from_scratch:
    ckpt_decoder.restore(mngr_decoder.latest_checkpoint)
    if mngr_decoder.latest_checkpoint:
      print('Decoder restored from %s\n' % (mngr_decoder.latest_checkpoint))
  if from_scratch or not mngr_decoder.latest_checkpoint:
    print('Decoder initialized from scratch\n')
  if not os.path.exists('./%s' % (wrapp.model_name,)):
    os.mkdir('./%s' % (wrapp.model_name,))

  # Training loop for the decoder part
  if decode_mode == 'normal':
    batch_maker = BatchMaker('decode', n_objs, batch_size, wrapp.n_frames, im_dims)
  elif decode_mode == 'sqm':
    batch_maker = BatchMaker('sqm', n_objs, batch_size, wrapp.n_frames, im_dims, 'V')
  for _ in range(n_epochs):
    e = ckpt_decoder.optim.iterations//n_batches

    # Train the decoder for one epoch
    mean_loss = 0.0
    mean_acc  = 0.0
    for b in range(n_batches):  # batch shape: (batch_s, n_frames) + im_dims
      batch, labels = batch_maker.generate_batch()
      acc, loss     = wrapp.train_step(tf.stack(batch, axis=1)/255, b, ckpt_decoder.optim, labels, -1)
      mean_loss    += loss
      mean_acc     += acc
      print('\r  Running batch %02i/%02i' % (b+1, n_batches), end='')

    # Record loss for this epoch
    mean_loss = mean_loss/n_batches
    mean_acc  = mean_acc /n_batches
    lr_str = "{:.2e}".format(ckpt_decoder.optim._decayed_lr(tf.float32).numpy())
    print('\nFinishing epoch %03i, lr = %s, accuracy = %.3f, loss = %.3f' % (e, lr_str, mean_acc, mean_loss))
    loss_tens = tf.concat([tf.zeros((e,)), mean_loss*tf.ones((1,)), tf.zeros((1000-e-1,))], axis=0)
    acc_tens  = tf.concat([tf.zeros((e,)), mean_acc *tf.ones((1,)), tf.zeros((1000-e-1,))], axis=0)
    ckpt_decoder.losses.assign_add(tf.Variable(loss_tens))
    ckpt_decoder.accs  .assign_add(tf.Variable( acc_tens))
    
    # Save checkpoint if necessary
    if e % 10 == 9:
      mngr_decoder.save()
      print('\nDecoder checkpoint saved at %s' % (mngr_decoder.latest_checkpoint,))

  # Plot loss and accuracy curves
  wrapp.plot_results(range(e+1), ckpt_decoder.losses.numpy()[:e+1], 'epoch', 'loss', 'decode')
  wrapp.plot_results(range(e+1), ckpt_decoder.  accs.numpy()[:e+1], 'epoch', 'acc' , 'decode')


if __name__ == '__main__':

  crit_type   = 'entropy_thresh'  # can be 'entropy', 'entropy_thresh', 'prediction_error', 'last_frame'
  decode_mode = 'sqm'             # can be 'normal' or 'sqm' (use 'V' sqm samples to train decoder)
  n_objs      = 2                 # number of moving object in each sample
  noise_lvl   = 0.1               # amount of noise added to the input (from 0.0 to 1.0)
  im_dims     = (64, 64, 3)       # image dimensions
  n_frames    = 13                # frames in the input sequences
  n_epochs    = 100               # epochs ran IN ADDITION TO latest checkpoint epoch
  batch_size  = 16                # sample sequences sent in parallel
  n_batches   = 64                # batches per epoch
  init_lr     = 1e-3              # first parameter to tune if does not work
  model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet'
  recons      = None
  decoder     = simple_decoder()
  wrapp       = Wrapper(model, recons, decoder, noise_lvl, crit_type, n_frames, name)
  train_decode(wrapp, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr, decode_mode, from_scratch=False)
