# Estimate the optimal initial learning rate

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1' # MacOS pb

from dataset import BatchMaker
from models import *


def find_best_lr(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, mode = 'decode'):

    batch_maker = BatchMaker(mode, obj_type, n_objs, batch_size, wrapp.n_frames, im_dims)

    # Learning devices
    lower_lr = 1e-5
    upper_lr = 1e-1
    lrs = np.linspace(lower_lr, upper_lr, n_epochs*n_batches)
    losses = []
    optim = tf.keras.optimizers.Adam(learning_rate=lower_lr)

    count = 0

    if mode == 'recons':
        for e in range(n_epochs):
            for b in range(n_batches):  # batch shape: (batch_s, n_frames) + im_dims
                batch = tf.stack(batch_maker.generate_batch(), axis=1)/255
                loss = wrapp.train_step(batch, b, e, optim)
                losses.append(loss)
                optim.learning_rate = lrs[count]
                count += 1
                print('\nBatch %02i, lr = %.5f, dec loss = %.3f' % (b, optim.learning_rate.numpy(), loss.numpy()))
    else:
        for e in range(n_epochs):
            for b in range(n_batches): # batch shape: (batch_s, n_frames) + im_dims
                batch, labels = batch_maker.generate_batch()
                loss = wrapp.train_step(tf.stack(batch, axis=1)/255, b, e, optim, labels)
                losses.append(loss)
                optim.learning_rate = lrs[count]
                count += 1
                print('\nBatch %s, lr = %.5f, dec loss = %.3f' % (b, tf.cast(optim.learning_rate, tf.float32).numpy(), loss.numpy()))
    
    # Plot the loss vs learning rate
    plt.figure()
    plt.ylabel("Loss")
    plt.xlabel("Learning rate (log scale)")
    plt.xscale("log")
    plt.plot(lrs, losses)
    plt.show()


if __name__ == '__main__':

  obj_type    = 'neil'       # can be 'ball' or 'neil' for now
  n_objs      = 2            # number of moving object in each sample
  im_dims     = (64, 64, 3)  # image dimensions
  n_frames    = 20           # frames in the input sequences
  n_epochs    = 50          # epochs ran after latest checkpoint epoch
  batch_size  = 16           # sample sequences sent in parallel
  n_batches   = 64           # batches per epoch
  model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet2'
  wrapp       = Wrapper(model, my_recons, my_decoder, n_frames, name)
  find_best_lr(wrapp, obj_type, n_objs, im_dims, n_epochs, batch_size, n_batches, 'decode')