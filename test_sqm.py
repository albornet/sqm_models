# Training procedure
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1' # MacOS pb
from dataset import BatchMaker
from models import *

def test_sqm(wrapp, obj_type, n_objs, im_dims, batch_size, n_batches, condition, n_tries):

  # Checkpoint (save and load model weights and accuracies)
  model_dir    = '%s/%s/ckpt_model' % (os.getcwd(), wrapp.model_name)
  decoder_dir  = '%s/%s/ckpt_decod' % (os.getcwd(), wrapp.model_name)
  ckpt_model   = tf.train.Checkpoint(net=wrapp.model)
  ckpt_decoder = tf.train.Checkpoint(net=wrapp.decoder)
  mngr_model   = tf.train.CheckpointManager(ckpt_model,   directory=model_dir,   max_to_keep=1)
  mngr_decoder = tf.train.CheckpointManager(ckpt_decoder, directory=decoder_dir, max_to_keep=1)

  # Try to load latest checkpoints 
  ckpt_model.restore(mngr_model.latest_checkpoint).expect_partial()
  if mngr_model.latest_checkpoint:
    print('\nReconstruction model loaded from %s\n' % (mngr_model.latest_checkpoint))
  else:
    print('\nWarning: your reconstruction model is not trained yet. Loading from scratch\n')
  ckpt_decoder.restore(mngr_decoder.latest_checkpoint)
  if mngr_decoder.latest_checkpoint:
    print('Decoder restored from %s\n' % (mngr_decoder.latest_checkpoint))
  else:
    print('Warning: your decoder is not trained yet. Loading from scratch\n')
  if not os.path.exists('./%s' % (wrapp.model_name,)):
    os.mkdir('./%s' % (wrapp.model_name,))

  # Test loop
  batch_maker  = BatchMaker('decode', obj_type, n_objs, batch_size, wrapp.n_frames, im_dims, condition)
  losses = np.zeros(n_tries)
  accs = np.zeros(n_tries)
  for try_ in range(n_tries):
    mean_loss = 0.0
    mean_acc  = 0.0
    for b in range(n_batches):  # batch shape: (batch_s, n_frames) + im_dims
      batch, labels = batch_maker.generate_batch()
      acc, loss  = wrapp.test_step(tf.stack(batch, axis=1)/255, labels, -1)
      mean_loss += loss
      mean_acc  += acc
    mean_loss    = mean_loss/n_batches
    mean_acc     = mean_acc /n_batches
    losses[try_] = mean_loss
    accs[try_]   = mean_acc
    print('\rRunning try %03i, mean batch accuracy = %.3f, mean batch loss = %.3f' % (try_, mean_acc, mean_loss), end='')
  
  print('\n  Mean try accuracy = %.3f, mean try loss = %.3f' % (np.mean(accs), np.mean(losses)))



if __name__ == '__main__':

  obj_type    = 'sqm'        # can be 'ball' or 'neil' or 'sqm' for now
  condition   = 'A-AV'       # can be 'V', 'V-AV' or 'V-PV'
  n_objs      = 2            # number of moving object in each sample
  im_dims     = (64, 64, 1)  # image dimensions
  n_frames    = 10           # frames in the input sequences
  n_tries     = 2            # number of tests
  batch_size  = 16           # sample sequences sent in parallel
  n_batches   = 4            # batches per try
  model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet2'
  decoder     = conv_decoder()
  wrapp       = Wrapper(model, my_recons, decoder, n_frames, name)
  test_sqm(wrapp, obj_type, n_objs, im_dims, batch_size, n_batches, condition, n_tries)