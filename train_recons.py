# Training procedure
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # avoid printing GPU info messages
from dataset import BatchMaker
from models import *

# Parameters
object_type = 'ball'       # can be 'ball', 'neil'
n_objects   = 3            # number of moving object in each sample
im_dims     = (64, 64, 3)  # image dimensions
n_channels  = im_dims[-1]  # color channels
n_frames    = 20           # frames in the input sequences
n_units     = 100          # neurons (not used by PredNet)
n_epochs    = 500          # epochs ran after latest checkpoint epoch
batch_size  = 16           # sample sequences sent in parallel
n_batches   = 64           # batches per epoch
initial_lr  = 2e-4         # first parameter to tune if does not work
batch_maker = BatchMaker('recons', object_type, n_objects, batch_size, n_frames, im_dims)

# Define network
model, model_name = PredNet((n_channels, 32, 64, 128), (n_channels, 32, 64, 128)), 'prednet2'
wrapp = Wrapper(model, my_recons, my_decoder, n_frames, model_name)
sched = tf.keras.experimental.CosineDecayRestarts(
  initial_learning_rate=initial_lr, first_decay_steps=10*n_batches, t_mul=2.0, m_mul=0.9, alpha=0.2)
optim = tf.keras.optimizers.Adam(sched)

# Checkpoint (save and load model weights)
model_dir  = '%s/%s/ckpt_model' % (os.getcwd(), model_name)
ckpt_model = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optim, net=model)
mngr_model = tf.train.CheckpointManager(ckpt_model, directory=model_dir, max_to_keep=1)
ckpt_model.restore(mngr_model.latest_checkpoint)
if mngr_model.latest_checkpoint:
  print('\nModel %s restored from %s' % (model_name, mngr_model.latest_checkpoint))
else:
  print('\nModel %s initialized from scratch' % (model_name))
  if not os.path.exists('./%s' % (model_name,)):
    os.mkdir('./%s' % (model_name,))

# Training loop for the reconstruction part
for e in range(n_epochs):
  if ckpt_model.step % 10 == 0 and e > 0:
    mngr_model.save()
    print('\nCheckpoint saved at %s' % (mngr_model.latest_checkpoint,))
  ckpt_model.step.assign_add(1)
  for b in range(n_batches):  # batch shape: (batch_s, n_frames) + im_dims
    batch = tf.stack(batch_maker.generate_batch(), axis=1)/255
    wrapp.train_step(batch, b, e, optim)
    print('\r  Running batch %02i/%2i' % (b+1, n_batches), end='')