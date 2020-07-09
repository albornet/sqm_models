# Import useful libraries and functions
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'   # avoid printing GPU info messages
os.environ['KMP_DUPLICATE_LIB_OK'] = '1'   # MacOS pb
# if os.getcwd() == '/content':            # COMMENT THE IF LOOP IS NOT ON COLAB
#   from google.colab import drive
#   drive.mount('/content/drive')
#   %cd /content/drive/My\ Drive/sqm_models
from dataset      import BatchMaker
from models       import *
from find_best_lr import find_best_lr
from train_recons import train_recons
from train_decode import train_decode
from test_sqm     import test_sqm

# Main parameters
im_dims     = (64, 64, 3)                                        # image dimensions
decode_crit = 'last_frame'                                       # can be 'entropy', 'entropy_thresh', 'pred_error', 'last_frame'
decode_mode = 'sqm'                                              # can be 'normal' or 'sqm' (use 'V' sqm samples to train decoder)
n_subjs_sqm = 10                                                 # number of subjects tested with the sqm paradigms (for stdevs)
batch_size  = {'recons': 16,      'decode': 16,    'sqm': 16  }  # sample sequences sent in parallel
n_batches   = {'recons': 64,      'decode': 64,    'sqm': 5   }  # batches per epoch (sqm: 80 = 5*16 trials per subject?)
n_frames    = {'recons': [8, 13], 'decode': 13,    'sqm': 13  }  # frames in the input sequences for reconstruction
n_epochs    = {'recons': 50,      'decode': 100,   'sqm': None}  # epochs ran after latest checkpoint epoch (for every frame number)
n_objs      = {'recons': 10,      'decode': 2,     'sqm': 2   }  # number of moving objects in recons batches
noise_lvl   = {'recons': 0.9,     'decode': 0.1,   'sqm': 0.1 }  # amount of noise added to reconstruction set samples
init_lr     = {'recons': 5e-4,    'decode': 1e-5,  'sqm': None}  # first parameter to tune if does not work
do_best_lr  = {'recons': False,   'decode': False, 'sqm': None}  # run (or not) find_best_lr to initiate learning rate
do_run      = {'recons': False,   'decode': False, 'sqm': True}  # run (or not) the training / testing

# Models and wrapper
model, name = PredNet((im_dims[-1], 32, 64, 128), (im_dims[-1], 32, 64, 128)), 'prednet'
recons      = None
decoder     = simple_decoder()
wrapp       = Wrapper(model, recons, decoder, 0.0, decode_crit, 0, name)

# Train model on next frame prediction
if do_run['recons']:
  wrapp.set_noise(noise_lvl['recons'])
  for n in n_frames['recons']:
    wrapp.n_frames  = n
    if do_best_lr['recons']:
      init_lr['recons'] = find_best_lr(wrapp, n_objs['recons'], im_dims, batch_size['recons'], mode='recons', custom=False, from_scratch=True)
    train_recons(wrapp, n_objs['recons'], im_dims, n_epochs['recons'], batch_size['recons'], n_batches['recons'], init_lr['recons'], from_scratch=False)

# Train decoder on vernier discrimination
if do_run['decode']:
  wrapp.n_frames = n_frames['decode']
  wrapp.set_noise(noise_lvl['decode'])
  if do_best_lr['decode']:
    init_lr['decode'] = find_best_lr(wrapp, n_objs['decode'], im_dims, batch_size['decode'], mode='decode', custom=False, from_scratch=True)
  train_decode(wrapp, n_objs['decode'], im_dims, n_epochs['decode'], batch_size['decode'], n_batches['decode'], init_lr['decode'], decode_mode, from_scratch=False)

# Test model on SQM paradigm
if do_run['sqm']:
  final_accuracies = {'V': [], 'P': [], 'A': []}
  final_stand_devs = {'V': [], 'P': [], 'A': []}
  wrapp.n_frames   = n_frames['sqm']
  wrapp.set_noise(noise_lvl['sqm'])
  plt.figure()
  plt.title('SQM results')
  for cond in final_accuracies.keys():
    if cond == 'V':
      mean, stdv, _, _ = test_sqm(wrapp, n_objs['sqm'], im_dims, batch_size['sqm'], n_batches['sqm'], n_subjs_sqm, cond)
      final_accuracies[cond].append(mean  )
      final_stand_devs[cond].append(stdv/2)  # above and below in plt.errorbar, so we divide by 2 
      plt.hlines(final_accuracies[cond], 0, n_frames['sqm']-4, colors='k', linestyles='dashed', label=cond)
    else:
      sec_frames = range(1, n_frames['sqm']-3)
      for sec_frame in sec_frames:
        this_cond = 'V-%sV%s' % (cond, sec_frame)
        mean, stdv, _, _ = test_sqm(wrapp, n_objs['sqm'], im_dims, batch_size['sqm'], n_batches['sqm'], n_subjs_sqm, this_cond)
        final_accuracies[cond].append(mean  )
        final_stand_devs[cond].append(stdv/2)  # above and below in plt.errorbar, so we divide by 2 
      plt.errorbar(sec_frames, final_accuracies[cond], final_stand_devs[cond], label=cond)
  plt.legend()
  plt.savefig('./%s/sqm_results.png' % (name))
  plt.show()