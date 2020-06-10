# Training procedure
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # avoid printing GPU info messages
#os.environ['KMP_DUPLICATE_LIB_OK'] = '1' # MacOS pb

from dataset import BatchMaker
from models import *

def train_decode(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr):

  # Learning devices
  sched = tf.keras.experimental.CosineDecayRestarts(
    initial_learning_rate=init_lr, first_decay_steps=10*n_batches,
    t_mul=2.0, m_mul=0.9, alpha=0.2)
  optim = tf.keras.optimizers.Adam(sched)

  # Checkpoint (save and load model weights)
  model_dir = '%s/%s/ckpt_model' % (os.getcwd(), wrapp.model_name)
  decoder_dir = '%s/%s/ckpt_decod' % (os.getcwd(), wrapp.model_name)
  ckpt_model = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optim, net=wrapp.model)
  ckpt_decoder = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optim, net=wrapp.decoder)
  mngr_model = tf.train.CheckpointManager(ckpt_model, directory=model_dir, max_to_keep=1)
  mngr_decoder = tf.train.CheckpointManager(ckpt_decoder, directory=decoder_dir, max_to_keep=1)
  ckpt_model.restore(mngr_model.latest_checkpoint)
  ckpt_decoder.restore(mngr_decoder.latest_checkpoint)
  if mngr_decoder.latest_checkpoint:
    print('\nDecoder restored from %s' % (mngr_decoder.latest_checkpoint))
  else:
    print('\nDecoder initialized from scratch')
  if not os.path.exists('./%s' % (wrapp.model_name,)):
   os.mkdir('./%s' % (wrapp.model_name,))

  # Training loop for the decoder part
  batch_maker = BatchMaker('decode', 'neil', n_objs, batch_size, n_frames, im_dims)
  for e in range(n_epochs):
    if ckpt_decoder.step % 10 == 0 and e > 0:
      mngr_decoder.save()
      print('\nCheckpoint saved at %s' % (mngr_decoder.latest_checkpoint,))
    ckpt_decoder.step.assign_add(1)
    for b in range(n_batches): # batch shape: (batch_s, n_frames) + im_dims
      batch, labels = batch_maker.generate_batch()
      wrapp.train_step(tf.stack(batch, axis=1)/255, b, e, optim, labels)
      print('\r Running batch %02i/%2i' % (b+1, n_batches), end='')


if __name__ == '__main__':

  obj_type = 'neil' # can be 'ball' or 'neil' for now
  n_objs = 2 # number of moving object in each sample
  im_dims = (64, 64, 3) # image dimensions
  n_frames = 20 # frames in the input sequences
  n_epochs = 200 # epochs ran after latest checkpoint epoch
  batch_size = 16 # sample sequences sent in parallel
  n_batches = 64 # batches per epoch
  init_lr = 2e-4 # first parameter to tune if does not work
  model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet2'
  wrapp = Wrapper(model, my_recons, my_decoder, n_frames, name)
  train_decode(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr)
