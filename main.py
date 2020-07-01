# Import useful libraries and functions
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'   # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1'   # MacOS pb
# from google.colab import drive           # UNCOMMENT THIS IN COLAB
# drive.mount('/content/drive')            # UNCOMMENT THIS IN COLAB
# %cd /content/drive/My\ Drive/sqm_models  # UNCOMMENT THIS IN COLAB
from dataset      import BatchMaker
from models       import *
from find_best_lr import find_best_lr
from train_recons import train_recons
from train_decode import train_decode
from test_sqm     import test_sqm

# Main parameters
im_dims          = (64, 64, 3)       # image dimensions
batch_size       = 16                # sample sequences sent in parallel
n_batches        = 64                # batches per epoch
crit_type        = 'entropy_thresh'  # can be 'entropy', 'entropy_thresh', 'pred_error'
n_frames_recons  = [8, 13, 20]       # frames in the input sequences for reconstruction
n_frames_decode  = 13                # frames in the input sequences for decoding
n_frames_sqm     = 13                # frames in the input sequences for the sqm paradigm
n_epochs_recons  = 20                # epochs ran after latest checkpoint epoch (for every frame number)
n_epochs_decode  = 50                # epochs ran after latest checkpoint epoch
n_objs_recons    = 10                # number of moving objects in recons batches
n_objs_decode    = 2                 # number of moving objects in decode batches
n_objs_sqm       = 2                 # number of moving objects in sqm paradigm
noise_lvl_recons = 0.8               # amount of noise added to reconstruction set samples
noise_lvl_decode = 0.8               # amount of noise added to decoding set samples
noise_lvl_sqm    = 0.1               # amount of noise added to sqm samples
init_lr_recons   = 2e-4              # first parameter to tune if does not work
init_lr_decode   = 1e-4              # first parameter to tune if does not work
do_train_recons  = True
do_train_decode  = True
do_test_sqm      = True
do_find_best_lr  = False

# Models and wrapper
model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet'
recons      = None
decoder     = simple_decoder()
wrapp       = wrapp = Wrapper(model, recons, decoder, 0.0, crit_type, 0, name)

# Train model on next frame prediction
if do_train_recons:
  wrapp.set_noise(noise_lvl_recons)
  for n in n_frames_recons:
    wrapp.n_frames  = n
    if do_find_best_lr:
      init_lr_recons = find_best_lr(wrapp, n_objs_recons, im_dims, batch_size, mode='recons', custom=False, from_scratch=True)
    train_recons(wrapp, n_objs_recons, im_dims, n_epochs_recons, batch_size, n_batches, init_lr_recons, from_scratch=False)

# Train decoder on vernier discrimination
if do_train_decode:
  wrapp.n_frames = n_frames_decode
  wrapp.set_noise(noise_lvl_decode)
  if do_find_best_lr:
    init_lr_decode = find_best_lr(wrapp, n_objs_decode, im_dims, batch_size, mode='decode', custom=False, from_scratch=True)
  train_decode(wrapp, n_objs_decode, im_dims, n_epochs_decode, batch_size, n_batches, init_lr_decode, from_scratch=False)

# Test model on SQM paradigm
if do_test_sqm:
  wrapp.n_frames = n_frames_sqm
  wrapp.set_noise(noise_lvl_sqm)
  test_sqm(wrapp, n_objs_sqm, im_dims, batch_size, n_batches, 'V', sqm_n_trials)
  for sec_frame in range(3, n_frames_sqm):
    test_sqm(wrapp, n_objs_sqm, im_dims, batch_size, n_batches, 'V-PV' + str(sec_frame), sqm_n_trials)
    test_sqm(wrapp, n_objs_sqm, im_dims, batch_size, n_batches, 'V-AV' + str(sec_frame), sqm_n_trials)
