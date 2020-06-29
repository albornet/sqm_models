# Training procedure
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1' # MacOS pb
from dataset import BatchMaker
from models import *

def test_sqm(wrapp, n_objs, im_dims, batch_size, n_batches, condition):

  # Checkpoint (save and load model weights and accuracies)
  model_dir    = '%s/%s/ckpt_model' % (os.getcwd(), wrapp.model_name)
  decoder_dir  = '%s/%s/ckpt_decod' % (os.getcwd(), wrapp.model_name)
  ckpt_model   = tf.train.Checkpoint(net=wrapp.model)
  ckpt_decoder = tf.train.Checkpoint(net=wrapp.decoder)
  mngr_model   = tf.train.CheckpointManager(ckpt_model,   directory=model_dir,   max_to_keep=1)
  mngr_decoder = tf.train.CheckpointManager(ckpt_decoder, directory=decoder_dir, max_to_keep=1)

  # Try to load latest checkpoints 
  ckpt_model  .restore(mngr_model  .latest_checkpoint).expect_partial()
  ckpt_decoder.restore(mngr_decoder.latest_checkpoint).expect_partial()
  if mngr_model.latest_checkpoint:
    print('\nReconstruction model loaded from %s\n' % (mngr_model.latest_checkpoint))
  else:
    print('\nWarning: your reconstruction model is not trained yet. Loading from scratch\n')
  if mngr_decoder.latest_checkpoint:
    print('Decoder restored from %s\n' % (mngr_decoder.latest_checkpoint))
  else:
    print('Warning: your decoder is not trained yet. Loading from scratch\n')
  if not os.path.exists('./%s' % (wrapp.model_name,)):
    os.mkdir('./%s' % (wrapp.model_name,))

  # Test loop
  batch_maker = BatchMaker('sqm', n_objs, batch_size, wrapp.n_frames, im_dims, condition)
  mean_loss   = 0.0
  mean_acc    = 0.0
  for b in range(n_batches):  # batch shape: (batch_s, n_frames) + im_dims
    batch, labels = batch_maker.generate_batch()
    acc, loss     = wrapp.test_step(tf.stack(batch, axis=1)/255, b, labels, -1)
    mean_loss    += loss
    mean_acc     += acc
    print('\r  Running batch %03i/%03i' % (b+1, n_batches), end='')
  mean_loss = mean_loss/n_batches
  mean_acc  = mean_acc /n_batches
  print('\nMean accuracy = %.3f, mean loss = %.3f' % (mean_acc, mean_loss))


if __name__ == '__main__':

  condition   = 'V'               # can be 'V', 'V-AV' or 'V-PV'
  crit_type   = 'entropy_thresh'  # can be 'entropy', 'entropy_threshold', 'prediction_error'
  n_objs      = 2                 # number of moving object in each sample
  noise_lvl   = 0.1               # amount of noise added to the input (from 0.0 to 1.0)
  im_dims     = (64, 64, 3)       # image dimensions
  n_frames    = 13                # frames in the input sequences
  batch_size  = 16                # sample sequences sent in parallel
  n_batches   = 64                # batches per try
  model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet'
  recons      = None
  decoder     = simple_decoder()
  wrapp       = Wrapper(model, recons, decoder, noise_lvl, crit_type, n_frames, name)
  test_sqm(wrapp, n_objs, im_dims, batch_size, n_batches, condition)