# Estimate optimal initial learning rate
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import math
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1'  # MacOS pb
from dataset import BatchMaker
from models  import *


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, n_samples, init_lr, stop_lr):
        super(CustomSchedule, self).__init__()
        init_exp = np.log10(init_lr)
        stop_exp = np.log10(stop_lr)
        self.predef_lrs = np.logspace(init_exp, stop_exp, n_samples)
    def __call__(self, step):
        return self.predef_lrs[int(step)-1]


def find_best_lr(wrapp, obj_type, n_objs, im_dims, batch_size, mode='decode', custom=True, from_scratch=False):
  
  # Simulation parameters
  n_samples = 200   # how many lrs are tried
  init_lr   = 1e-7  # smallest lr tried
  stop_lr   = 1e-0  # largest lr tried

  # Learning devices
  if custom:
    scheduler = CustomSchedule(n_samples, init_lr, stop_lr)
  else:
    decay_rate = stop_lr/init_lr
    scheduler  = tf.keras.optimizers.schedules.ExponentialDecay(
    init_lr, n_samples, decay_rate, staircase=False, name=None)
  optim = tf.keras.optimizers.Adam(scheduler)

  # Run batches with increasing learning rates 
  batch_maker = BatchMaker(mode, obj_type, n_objs, batch_size, wrapp.n_frames, im_dims)
  lrs         = []
  losses      = []

  ######### Pre train over 10 epochs ########
  print("Pre train over 10 epochs")
  sched_pretraining = tf.keras.experimental.CosineDecayRestarts(
    initial_learning_rate=2e-4, first_decay_steps=10*64,
    t_mul=2.0, m_mul=0.9, alpha=0.2)
  optim_pretraining = tf.keras.optimizers.Adam(sched_pretraining)

  for e in range(10):
    for b in range(64):  # batch shape: (batch_s, n_frames) + im_dims
      batch = tf.stack(batch_maker.generate_batch(), axis=1)/255
      rec_loss = wrapp.train_step(batch, b, e, optim_pretraining)
      if b == 0:
        lr_str = "{:.2e}".format(optim_pretraining._decayed_lr(tf.float32).numpy())
        print('\nStarting epoch %03i, lr = %s, rec loss = %.3f' % (e, lr_str, rec_loss))
      print('\r  Running batch %02i/%2i' % (b+1, 64), end='')
  ############################################  

  for s in range(n_samples):

    # Compute loss
    if mode == 'recons':
      batch = batch_maker.generate_batch()
      loss = wrapp.train_step(tf.stack(batch, axis=1)/255, s, optim)
    elif mode == 'decode':
      batch, labels = batch_maker.generate_batch()
      loss = wrapp.train_step(tf.stack(batch, axis=1)/255, s, optim, labels)
    
    # Record loss
    losses.append(loss.numpy())
    updated_lr = optim._decayed_lr(tf.float32).numpy()
    lrs.append(updated_lr)
    lr_str = "{:.2e}".format(updated_lr)
    print('\rSample %03i/%03i, lr = %s, loss = %.4f' % (s+1, n_samples, lr_str, loss), end='')

  # Figure out best lr and plot loss vs learning rate
  lr_for_min_loss = lrs[np.argmin(losses)]
  lr_opt          = lr_for_min_loss/10
  print('\nMin loss for lr = %s, opt lr = %s' %(lr_for_min_loss, lr_opt)) 
  plot_output(lrs, losses, wrapp.model_name)

  # Return best lr
  return lr_opt


def plot_output(lrs, losses, name):
    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Learning rate (log scale)")
    plt.xscale("log")
    plt.plot(lrs, losses)
    plt.grid()
    plt.savefig('./%s/loss_vs_lr.png' % (name))
    plt.show()
    plt.close()


if __name__ == '__main__':

  obj_type    = 'neil'       # can be 'ball' or 'neil' for now
  train_mode  = 'decode'     # can be 'recons' or 'decode'
  n_objs      = 2            # number of moving object in each sample
  im_dims     = (64, 64, 1)  # image dimensions
  n_frames    = 20           # frames in the input sequences
  batch_size  = 16           # sample sequences sent in parallel
  model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet2'
  wrapp       = Wrapper(model, my_recons, my_decoder, n_frames, name)
  init_lr     = find_best_lr(wrapp, obj_type, n_objs, im_dims, batch_size,
                             mode=train_mode, custom=False, from_scratch=True)