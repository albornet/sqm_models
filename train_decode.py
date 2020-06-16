# Training procedure
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1' # MacOS pb
from dataset import BatchMaker
from models import *


def train_decode(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr):

  # Learning devices
  sched = tf.keras.experimental.CosineDecayRestarts(
    initial_learning_rate=init_lr, first_decay_steps=n_batches, t_mul=2.0, m_mul=0.95, alpha=0.3)
  optim = tf.keras.optimizers.Adam(sched)

  # To store the losses and accuracies
  losses = tf.Variable(tf.zeros((1000,1)))
  accs = tf.Variable(tf.zeros((1000,1)))
  # Checkpoint (save and load model weights and results)
  model_dir    = '%s/%s/ckpt_model' % (os.getcwd(), wrapp.model_name)
  decoder_dir  = '%s/%s/ckpt_decod' % (os.getcwd(), wrapp.model_name)
  ckpt_model   = tf.train.Checkpoint(net=wrapp.model)
  ckpt_decoder = tf.train.Checkpoint(optimizer=optim, net=wrapp.decoder, loss=losses, acc=accs)
  mngr_model   = tf.train.CheckpointManager(ckpt_model,   directory=model_dir,   max_to_keep=1)
  mngr_decoder = tf.train.CheckpointManager(ckpt_decoder, directory=decoder_dir, max_to_keep=1)
  ckpt_model  .restore(mngr_model  .latest_checkpoint)
  ckpt_decoder.restore(mngr_decoder.latest_checkpoint)
  if mngr_model.latest_checkpoint:
    print('\nReconstruction model loaded from %s' % (mngr_decoder.latest_checkpoint))
  else:
    print('\nWarning: your reconstruction model is not trained yet. Loading from scratch.')
  if mngr_decoder.latest_checkpoint:
    print('\nDecoder restored from %s' % (mngr_decoder.latest_checkpoint))
  else:
    print('\nDecoder initialized from scratch')
  if not os.path.exists('./%s' % (wrapp.model_name,)):
   os.mkdir('./%s' % (wrapp.model_name,))

  # Training loop for the decoder part
  batch_maker = BatchMaker('decode', 'neil', n_objs, batch_size, wrapp.n_frames, im_dims)
  for e in range(n_epochs):
    # Mean batch loss and mean batch accuracy
    mean_loss = 0.0
    mean_acc = 0.0
    loss_tens = np.zeros((1000,1))
    acc_tens = np.zeros((1000,1))
    opt_e = optim.iterations/n_batches
    if opt_e % 10 == 0 and e > 0:
      mngr_decoder.save()
      print('\nCheckpoint saved at %s' % (mngr_decoder.latest_checkpoint,))
    for b in range(n_batches):  # batch shape: (batch_s, n_frames) + im_dims
      batch, labels = batch_maker.generate_batch()
      acc, dec_loss = wrapp.train_step(tf.stack(batch, axis=1)/255, b, optim, labels, -1)
      mean_loss += dec_loss
      mean_acc += acc
      if b == 0:    
        lr_str = "{:.2e}".format(optim._decayed_lr(tf.float32).numpy())
        print('\nStarting epoch %03i, lr = %s, accuracy = %.3f, dec loss = %.3f' % (e, lr_str, acc, dec_loss))
      print('\r Running batch %02i/%2i' % (b+1, n_batches), end='')
    loss_tens[e,:] = mean_loss/n_batches
    acc_tens[e,:] = mean_acc/n_batches
    # Mean epoch loss and accuracy  
    losses.assign_add(tf.Variable(tf.cast(loss_tens, tf.float32)))
    accs.assign_add(tf.Variable(tf.cast(acc_tens, tf.float32)))

  # Plot the accuracy vs epochs
  plot_accuracy(tf.cast(optim.iterations, tf.float32).numpy()/n_batches, accs, wrapp.model_name)

  # Plot the loss vs epochs
  plot_loss(tf.cast(optim.iterations, tf.float32).numpy()/n_batches, accs, wrapp.model_name)
  

def plot_accuracy(epochs, accs, name):
    plt.figure()
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.plot(range(0, int(epochs)), tf.squeeze(accs)[:int(epochs)])
    plt.grid()
    plt.savefig('./%s/accuracy_vs_epoch_decoder.png' % (name))
    plt.show()
    plt.close()

def plot_loss(epochs, losses, name):
    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.plot(range(0, int(epochs)), tf.squeeze(losses)[:int(epochs)])
    plt.grid()
    plt.savefig('./%s/loss_vs_epoch_decoder.png' % (name))
    plt.show()
    plt.close()



if __name__ == '__main__':

  obj_type    = 'neil'       # can be 'ball' or 'neil' for now
  n_objs      = 2            # number of moving object in each sample
  im_dims     = (64, 64, 1)  # image dimensions
  n_frames    = 10           # frames in the input sequences
  n_epochs    = 20          # epochs ran after latest checkpoint epoch
  batch_size  = 16           # sample sequences sent in parallel
  n_batches   = 2           # batches per epoch
  init_lr     = 3.0e-5         # first parameter to tune if does not work
  model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet2'
  wrapp       = Wrapper(model, my_recons, my_decoder, n_frames, name)
  train_decode(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, init_lr)
