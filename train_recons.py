# Training procedure
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1' # MacOS pb
from dataset import BatchMaker
from models import *


def train_recons(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr, from_scratch=False):

  # Learning devices
  sched = tf.keras.experimental.CosineDecayRestarts(
    initial_learning_rate=init_lr, first_decay_steps=10*n_batches,
    t_mul=2.0, m_mul=0.9, alpha=0.2)
  optim = tf.keras.optimizers.Adam(sched)

  # Checkpoint (save and load model weights)
  model_dir  = '%s/%s/ckpt_model' % (os.getcwd(), wrapp.model_name)
  ckpt_model = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optim, net=wrapp.model)
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
    if ckpt_model.step % 10 == 0 and e > 0:
      mngr_model.save()
      print('\nCheckpoint saved at %s' % (mngr_model.latest_checkpoint,))
    ckpt_model.step.assign_add(1)
    for b in range(n_batches):  # batch shape: (batch_s, n_frames) + im_dims
      batch = tf.stack(batch_maker.generate_batch(), axis=1)/255
      rec_loss = wrapp.train_step(batch, b, e, optim)
      if b == 0:
        lr_str = "{:.2e}".format(optim._decayed_lr(tf.float32).numpy())
        print('\nStarting epoch %03i, lr = %s, rec loss = %.3f' % (e, lr_str, rec_loss))
      print('\r  Running batch %02i/%2i' % (b+1, n_batches), end='')


if __name__ == '__main__':

  obj_type    = 'neil'       # can be 'ball' or 'neil' for now
  n_objs      = 2            # number of moving object in each sample
  im_dims     = (64, 64, 3)  # image dimensions
  n_frames    = 20           # frames in the input sequences
  n_epochs    = 200          # epochs ran after latest checkpoint epoch
  batch_size  = 16           # sample sequences sent in parallel
  n_batches   = 64           # batches per epoch
  init_lr     = 2e-4         # first parameter to tune if does not work
  model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet2'
  wrapp       = Wrapper(model, my_recons, my_decoder, n_frames, name)
  train_recons(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr, from_scratch=True)
