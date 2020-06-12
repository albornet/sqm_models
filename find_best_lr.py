# Estimate the optimal initial learning rate

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler 
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import math
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1' # MacOS pb

from dataset import BatchMaker
from models import *


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, n_epochs, n_batches, init_lr = 1e-8, end_lr = 0.5):
        super(CustomSchedule, self).__init__()
        self.predefined_lr = [init_lr * math.exp(step * math.log(end_lr/init_lr)/(n_epochs*n_batches)) for step in range(n_epochs*n_batches)]

    def __call__(self, step):
        return self.predefined_lr[int(step)-1]

def find_best_lr(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, mode = 'decode', custom = True):

    batch_maker = BatchMaker(mode, obj_type, n_objs, batch_size, wrapp.n_frames, im_dims)
    
    # Learning devices
    init_lr = 1e-8
    end_lr = 0.5
    if custom:
        sched = CustomSchedule(n_epochs, n_batches, init_lr, end_lr)
    else:
        decay_steps = n_epochs*n_batches
        decay_rate = 1.1
        sched = tf.keras.optimizers.schedules.ExponentialDecay(
        init_lr, decay_steps, decay_rate, staircase=False, name=None)

    optim = tf.keras.optimizers.Adam(sched)

    lrs = []
    losses = []

    if mode == 'recons':
        for e in range(n_epochs):
            for b in range(n_batches):  # batch shape: (batch_s, n_frames) + im_dims
                batch = tf.stack(batch_maker.generate_batch(), axis=1)/255
                loss = wrapp.train_step(batch, b, e, optim)
                losses.append(loss.numpy())
                updated_lr = optim._decayed_lr(tf.float32).numpy()
                lrs.append(updated_lr)
                print('\nBatch %02i, lr = %s, dec loss = %.4f' % (b, updated_lr, loss))

    else:
        for e in range(n_epochs):
            for b in range(n_batches): # batch shape: (batch_s, n_frames) + im_dims
                batch, labels = batch_maker.generate_batch()
                loss = wrapp.train_step(tf.stack(batch, axis=1)/255, b, e, optim, labels)
                losses.append(loss.numpy())
                updated_lr = optim._decayed_lr(tf.float32).numpy()
                lrs.append(updated_lr)
                print('\nBatch %02i, lr = %s, dec loss = %.4f' % (b, updated_lr, loss))

    lr_for_min_loss = lrs[np.argmin(losses)]
    lr_opt = lrs[np.argmin(losses)]/10
    print('Min loss for lr = %s, opt lr = %s' %(lr_for_min_loss, lr_opt))
    
    # Plot the loss vs learning rate
    plot_output(lrs, losses, wrapp.model_name)

    return lr_opt

def plot_output(lrs, losses, name):
    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Learning rate (log scale)")
    plt.xscale("log")
    plt.plot(lrs, losses)
    plt.grid()
    plt.savefig('./%s/loss_vs_lr.png' % (name))
    plt.close()



if __name__ == '__main__':

  obj_type    = 'neil'       # can be 'ball' or 'neil' for now
  n_objs      = 2            # number of moving object in each sample
  im_dims     = (64, 64, 3)  # image dimensions
  n_frames    = 20           # frames in the input sequences
  n_epochs    = 50          # epochs ran after latest checkpoint epoch
  batch_size  = 16           # sample sequences sent in parallel
  n_batches   = 4           # batches per epoch
  model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet2'
  wrapp       = Wrapper(model, my_recons, my_decoder, n_frames, name)
  find_best_lr(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, 'recons', custom=True)