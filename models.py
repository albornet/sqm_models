import tensorflow as tf
import numpy as np
import scipy
import matplotlib.pyplot as plt
import imageio
import numpy as np


######################
### Reconstructors ###
######################

# Simple "deconvolution" reconstructor
my_recons = tf.keras.Sequential(
    [tf.keras.layers.Dense(4*8*32),
     tf.keras.layers.Reshape((4,8,32)),
     tf.keras.layers.Conv2DTranspose(16, (5,5), strides=(1,1), padding='same', activation='relu'),
     tf.keras.layers.Conv2DTranspose( 8, (5,5), strides=(2,2), padding='same', activation='relu'),
     tf.keras.layers.Conv2DTranspose( 4, (5,5), strides=(2,2), padding='same', activation='relu'),
     tf.keras.layers.Conv2DTranspose( 1, (5,5), strides=(2,2), padding='same', activation='relu')])

################
### Decoders ###
################

# Simple linear decoder
def simple_decoder():
  return tf.keras.Sequential(
    [tf.keras.layers.Flatten(),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dense(512, activation='relu'   ),
     tf.keras.layers.Dense(  2, activation='softmax')])

# Simple fully convolutional decoder
def my_fully_conv_decoder():
  return tf.keras.Sequential(
    [tf.keras.layers.Conv2D(512, (1,1), strides=(1,1), padding='same', activation='relu'),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Conv2D(2, (1,1), strides=(1,1), padding='same', activation='relu'),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.GlobalAveragePooling2D(),
     tf.keras.layers.Softmax()])

# All-convolutional decoder 
def all_cnn_decoder():
  return tf.keras.Sequential(
    [tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Conv2D(128, (2,2), strides=(2,2), padding='same', activation='relu'),
     tf.keras.layers.Conv2D(32, (2,2), strides=(2,2), padding='same', activation='relu'),
     tf.keras.layers.Conv2D(8, (2,2), strides=(2,2), padding='same', activation='relu'),
     tf.keras.layers.Conv2D(2, (2,2), strides=(2,2), padding='same', activation='softmax')])

# My try
def conv_decoder():
  return tf.keras.Sequential(
    [tf.keras.layers.Conv2D(32, (2,2), strides=(2,2), padding='same', activation='relu'   ),
     tf.keras.layers.Conv2D( 8, (2,2), strides=(2,2), padding='same', activation='relu'   ),
     tf.keras.layers.Conv2D( 2, (2,2), strides=(2,2), padding='same', activation='softmax')])

###################
### Core models ###
###################

# Connected network of simple rate cells
def simple_RNN(im_dims, n_frames, n_units):
  return tf.keras.Sequential([
         tf.keras.layers.InputLayer(input_shape=(n_frames,) + im_dims),
         tf.keras.layers.Reshape((n_frames, tf.math.reduce_prod(im_dims))),
         tf.keras.layers.SimpleRNN(units=n_units, return_sequences=True)])  # w/o return_seq: returns output only at last step (= last frame)


# Connected network of LSTM units
def simple_LSTM(im_dims, n_frames, n_units):
  return tf.keras.Sequential([
         tf.keras.layers.InputLayer(input_shape=(n_frames,) + im_dims),
         tf.keras.layers.Reshape((n_frames, tf.math.reduce_prod(im_dims))),
         tf.keras.layers.LSTM(units=n_units, return_sequences=True)])  # w/o return_seq: returns output only at last step (= last frame)


# Connected network of GRU units
def simple_GRU(im_dims, n_frames, n_units):
  return tf.keras.Sequential([
         tf.keras.layers.InputLayer(input_shape=(n_frames,) + im_dims),
         tf.keras.layers.Reshape((n_frames, tf.math.reduce_prod(im_dims))),
         tf.keras.layers.GRU(units=n_units, return_sequences=True)])  # w/o return_seq: returns output only at last step (= last frame)


# Convolutional layer that integrates information through space and time
def conv2D_LSTM(im_dims, n_frames, n_units):
  return tf.keras.Sequential([
         tf.keras.layers.InputLayer(input_shape=(n_frames,) + im_dims),
         # tf.keras.layers.Reshape((n_frames, tf.math.reduce_prod(im_dims))),
         tf.keras.layers.ConvLSTM2D(filters=16, kernel_size=(16,16), strides = (2,2), return_sequences=True, stateful=False, padding='same'),
         tf.keras.layers.Reshape((n_frames, -1))])


# Predictive coding network
class PredNet(tf.keras.Model):
  
  def __init__(self,  R_channels, A_channels, t_extrapolate=float('inf')):
  
    super(PredNet, self).__init__()
    self.r_channels    = R_channels + (0,)  # for convenience (last layer)
    self.a_channels    = A_channels
    self.n_layers      = len(R_channels)
    self.t_extrapolate = t_extrapolate

    for i in range(self.n_layers):  # number of input features: 2*self.a_channels[i] + self.r_channels[i+1]
      cell = tf.keras.layers.ConvLSTM2D(filters=self.r_channels[i], kernel_size=(3,3), return_sequences=True, stateful=True, padding='same')
      setattr(self, 'cell{}'.format(i), cell)

    # for i in range(self.n_layers):
    #   conv = tf.keras.layers.Conv2D(filters=self.a_channels[i], kernel_size=(3,3), padding='same', activation='relu')
    #   if i == 0:
    #     conv = tf.keras.Sequential([conv, SatLU()])
    #   setattr(self, 'conv{}'.format(i), conv)
    for i in range(self.n_layers):
      conv = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=self.a_channels[i], kernel_size=(3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization()])
      if i == 0:
        conv = tf.keras.Sequential([
          tf.keras.Sequential([conv, SatLU()]),
          tf.keras.layers.BatchNormalization()])
      setattr(self, 'conv{}'.format(i), conv)

    self.upsample = tf.keras.layers.UpSampling2D(size=(2,2))
    self.maxpool  = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2))
    for l in range(self.n_layers - 1):
      update_A = tf.keras.Sequential([tf.keras.layers.Conv2D(self.a_channels[l+1], (3, 3), padding='same'), self.maxpool])
      setattr(self, 'update_A{}'.format(l), update_A)

  def set_t_extrapolate(self, t):
    self.t_extrapolate = t

  def call(self, x):

    R_seq = [None]*self.n_layers
    H_seq = [None]*self.n_layers
    E_seq = [None]*self.n_layers
    state = [None]*self.n_layers

    h, w       = x.shape[-3], x.shape[-2]
    batch_size = x.shape[0]
    time_steps = x.shape[1]

    for l in range(self.n_layers):
      E_seq[l] = tf.zeros((batch_size,    h, w, 2*self.a_channels[l]))
      R_seq[l] = tf.zeros((batch_size, 1, h, w, 1*self.r_channels[l]))
      state[l] = tf.zeros((self.r_channels[l], self.r_channels[l]))
      w = w//2
      h = h//2
    
    frame_predictions = [[] for l in range(self.n_layers)]
    for t in range(time_steps):

      # Top-down pass to update LSTM states (R is LSTM neural state)
      for l in reversed(range(self.n_layers)):
        cell = getattr(self, 'cell{}'.format(l))
        if t == 0:
          try:
            cell.reset_states()  # first time step does not want last state of previous sequence
          except TypeError:
            pass
        E    = tf.expand_dims(E_seq[l], axis=1)
        R    = R_seq[l]
        if l == self.n_layers - 1:
          R = cell(E)
        else:
          R = cell(tf.concat([E, tf.expand_dims(self.upsample(tf.squeeze(R_seq[l+1], axis=1)), axis=1)], axis=-1))
        R_seq[l] = R

      # Bottom-up pass to compute predictions A_hat and prediction errors (A_hat - A)
      A = frame_predictions[0][-1] if t >= self.t_extrapolate else x[:,t]  # extrapolation makes forward zero
      for l in range(self.n_layers):
        conv     = getattr(self, 'conv{}'.format(l))
        A_hat    = conv(tf.squeeze(R_seq[l], axis=1))
        frame_predictions[l].append(A_hat)
        pos      = tf.nn.relu(A_hat - A)
        neg      = tf.nn.relu(A - A_hat)
        E        = tf.concat([pos, neg], axis=-1)
        E_seq[l] = E
        if l < self.n_layers - 1:
          update_A = getattr(self, 'update_A{}'.format(l))
          A        = update_A(E)
    
    return [tf.stack(frame_predictions[l], axis=1) for l in range(self.n_layers)]

# Helper function for prednet
class SatLU(tf.keras.Model):
    def __init__(self, lower=0, upper=1):
        super(SatLU, self).__init__()
        self.lower = lower
        self.upper = upper
    def call(self, input):
        return tf.clip_by_value(input, self.lower, self.upper)


####################
### All together ###
####################

# Wrapper class to combine core model, reconstructor and decoder
class Wrapper(tf.keras.Model):

  def __init__(self, model, reconstructor, decoder, crit_type, n_frames, model_name):
    super(Wrapper, self).__init__()
    self.model         = model
    self.add_noise     = tf.keras.layers.GaussianNoise(0.1)
    self.reconstructor = reconstructor
    self.decoder       = decoder
    self.crit_type     = crit_type
    self.n_frames      = n_frames
    self.model_name    = model_name
    self.accuracy      = tf.keras.metrics.Accuracy()

  def get_reconstructions(self, x):
    if isinstance(self.model, PredNet):
        return self.model(x)[0]
    else:
      # x = tf.cast(x, tf.float32)
      states = self.model(x)
      recs   = []
      for t in range(self.n_frames):
        recs.append(self.reconstructor(states[:,t]))
      return tf.stack(recs, axis=1)
  
  def compute_rec_loss(self, x):
    recons  = self.get_reconstructions(self.add_noise(x, True))
    weights = [1.0/(n+1) for n in range(self.n_frames-1)]  # first frame cannot be predicted
    if isinstance(self.model, PredNet):                    # nth PredNet output predicts nth frame
      if self.model.t_extrapolate < float('inf'):
        weights = [w if n < self.model.t_extrapolate else 2.0*w for n, w in enumerate(weights)]
      losses = [w*tf.reduce_sum((recons[:, n+1] - x[:, n+1])**2) for n, w in enumerate(weights)]
    else:
      losses = [w*tf.reduce_sum((recons[:, n] - x[:, n+1])**2) for n, w in enumerate(weights)]
    frame_losses = [w*tf.reduce_sum((x[:, n] - x[:, n+1])**2) for n, w in enumerate(weights)]
    return tf.reduce_sum(losses)/tf.reduce_sum(frame_losses)

  """
  def compute_dec_loss(self, labels, lat_var, criterion):
    loss_func = tf.keras.losses.BinaryCrossentropy()  # for the moment
    targets   = tf.one_hot(labels, 2)                 # Crossentropy needs one_hot
    weights   = [0.0 if n == 0 else 1.0 for n in range(self.n_frames)]
    losses    = tf.zeros(labels.shape)
    accurs    = 0.0
    for t in range(self.n_frames):
      decoding = tf.squeeze(self.decoder(lat_var[:, t]))
      losses  += weights[t]*loss_func(targets, decoding)
      self.accuracy.update_state(labels, tf.argmax(decoding, 1))
      accurs  += self.accuracy.result()
    self.accuracy.reset_states()
    return accurs/self.n_frames, tf.reduce_sum(losses)/self.n_frames
  """
  
  def compute_dec_loss(self, labels, lat_var, criterion):
      loss_func = tf.keras.losses.BinaryCrossentropy()  # for the moment
      targets   = tf.one_hot(labels, 2)                 # Crossentropy needs one_hot
      #weights   = [0.0 if n == 0 else 1.0 for n in range(self.n_frames)]

      # First look for the "first" smooth region of the criterion curve 
      variations                 = np.abs(np.diff(criterion[:, 1:], axis=-1))
      thresh                     = 0.1*np.amax(variations)
      variations_under_thresh    = (variations < thresh).astype(int)
      window_size                = 3   # here look at a window of 3 frames
      filter_                    = np.ones((1,window_size)) 
      stable_frames = scipy.signal.convolve2d(variations_under_thresh, filter_, mode='same')
      first_stable_frames        = np.ones(shape=criterion.shape[0], dtype=int)
      for b in range(criterion.shape[0]):
        for t in range(1, self.n_frames-2):
          if stable_frames[b, t] == window_size:
            first_stable_frames[b] += t
            break
      first_stable_frames[first_stable_frames == 1] = self.n_frames-window_size-1
      
      # Then look for the minimum of the derivative of the criterion in those regions
      frames_to_decode = np.zeros(criterion.shape[0])
      for b in range(criterion.shape[0]):
        frames_to_decode[b] =   first_stable_frames[b] + np.argmin(np.abs(np.diff(criterion[b, first_stable_frames[b]:first_stable_frames[b]+window_size])))
      print("  Frames to decode: ", frames_to_decode)
      
      # Derived from the previous criterias, select the frame to send to the decoder for each sample of the batch
      indices_to_decode = tf.stack([tf.range(lat_var.shape[0]), frames_to_decode], axis=-1)
      lat_var_to_decode = tf.gather_nd(lat_var, indices_to_decode)

      # Decode and compute the loss & accuracy
      decoding = tf.squeeze(self.decoder(lat_var_to_decode))
      loss     = loss_func(targets, decoding)
      self.accuracy.update_state(labels, tf.argmax(decoding, 1))
      accur    = self.accuracy.result()
      self.accuracy.reset_states()
      return accur, loss
  
  def compute_criterion(self, lat_var):
    batch_size = lat_var.shape[0]
    criterion  = np.zeros((batch_size, self.n_frames))
    if 'entropy' in self.crit_type:
      base_change = np.log(2.0)
      epsilon     = np.finfo(float).eps
      hist_range  = (0.03, 6.0) if 'thresh' in self.crit_type else (-0.1, 6.0)
      for b in range(batch_size):
        for t in range(1, self.n_frames):
          flat_var        = tf.keras.backend.flatten(lat_var[b, t]).numpy()
          probs, _        = np.histogram(flat_var, bins=100, range=hist_range, density=True)
          criterion[b, t] = -(probs*np.log(probs + epsilon)/base_change).sum()
    elif self.crit_type == 'pred_error':
      criterion = lat_var.numpy().sum(axis=(-1, -2, -3))  # sum over both space and feature dims
    return criterion

  def train_step(self, x, b, opt, labels=None, layer_decod=-1):

    # Run and record decoding
    if labels is not None:
      with tf.GradientTape() as tape:
        lat_var   = self.model(x)[layer_decod]
        criterion = self.compute_criterion(lat_var)
        acc, loss = self.compute_dec_loss(labels, lat_var, criterion)  # this should take the criterion into account!
      to_train  = self.decoder.trainable_variables
      if b == 0:
        self.plot_recons(x, sample_indexes=[0])
        self.plot_results(range(1, self.n_frames), criterion[0, 1:],
             'frame', 'criterion (%s)' % (self.crit_type), 'decode')
        self.plot_distrubution_activities_lat_vars(x, show=False)
        # self.plot_state_all_layers(x)
        # self.plot_state_layer(x)
      
    # Run and record reconstruction
    else:
      with tf.GradientTape() as tape:
        loss = self.compute_rec_loss(x)
      if isinstance(self.model, PredNet):  # Prednet generates reconstructions itself
        to_train = self.model.trainable_variables
      else:
        to_train = self.model.trainable_variables + self.reconstructor.trainable_variables
      if b == 0:
        self.plot_recons(x, sample_indexes=[0])
    
    # Apply gradient descent and return results for monitoring
    grad = tape.gradient(loss, to_train)
    opt.apply_gradients(zip(grad, to_train))

    if labels is not None:
      return acc, loss
    else:
      return loss
    
  def test_step(self, x, labels, layer_decod=-1):
    lat_var   = self.model(x)[layer_decod]
    acc, loss = self.compute_dec_loss(labels, lat_var)
    return acc, loss

  def plot_recons(self, x, sample_indexes, show=False, noisy=True):
    if noisy:
      x = self.add_noise(x, True)
    r  = tf.clip_by_value(self.get_reconstructions(x), 0.0, 1.0)
    t  = tf.zeros((10 if i == 3 else x.shape[i] for i in range(len(x.shape))))  # black rectangle
    xr = tf.clip_by_value(tf.concat((x, t, r), axis=3), 0.0, 1.0)
    for s in sample_indexes:
      f  = plt.figure(figsize=(int(self.n_frames*(x.shape[3] + 3)/32),int(2*(x.shape[2] + 3)/32)))
      for t in range(self.n_frames):
        ax = f.add_subplot(self.n_frames+1, 0*(self.n_frames+1) + t + 1, 1)
        ax.imshow(tf.squeeze(xr[s, t]), cmap='Greys')  # squeeze and cmap only apply to n_channels = 1
      if show:
        plt.show()
      plt.savefig('./%s/latest_input_vs_prediction_epoch_%02i.png' % (self.model_name, s))
      plt.close()
      xr_frames = [tf.cast(255*xr[s, t], tf.uint8).numpy() for t in range(self.n_frames)]
      imageio.mimsave('./%s/latest_input_vs_prediction_%02i.gif' % (self.model_name, s), xr_frames, duration=0.1)
  
  def plot_results(self, x_vals, y_vals, x_val_name, y_val_name, mode, show=True):
    plt.figure()
    plt.ylabel(y_val_name)
    plt.xlabel(x_val_name)
    if x_vals[-1] > 1e3*x_vals[1]:
      plt.xscale('log')
    plt.plot(x_vals, y_vals)
    plt.grid()
    plt.savefig('./%s/%s_%s_vs_%s.png' % (self.model_name, mode, y_val_name, x_val_name))
    if show:
      plt.show()
    plt.close()

  def plot_distrubution_activities_lat_vars(self, x, layer=-1, show=False):
    fig, axes = plt.subplots(self.n_frames, 1, figsize=(24, 24))
    for t in range(1, self.n_frames):
      flat_lat_vars  = tf.keras.backend.flatten(self.model(x)[layer][0, t]).numpy()
      axes[t].hist(flat_lat_vars, bins=100, range=(0.03, 6.0), density=True) 
      axes[t].set(xlabel = 'Values of neuron activities at frame ' + str(t+1), ylabel = 'Occurence')
      axes[t].grid()
    fig.savefig('./%s/distribution_of_neuron_activities.png' % (self.model_name))
    if show:
      fig.show()
    plt.close()

  def plot_state_all_layers(self, x, show=False):
    fig, axes = plt.subplots(self.model.n_layers, 2, figsize=(16, self.model.n_layers * 4))
    lat_vars  = self.model(x)
    for layer in range(self.model.n_layers): # plot aussi les autres layers juste pour voir, même si on s'en fout un peu
      first_sample = np.mean(lat_vars[layer][0, :, :, :, :], axis=-1) # average over channels
      if layer > 0:
        axes[layer, 0].plot(range(self.n_frames), 100*first_sample[:, :, 0]) 
        axes[layer, 0].set(xlabel = 'Frame', ylabel = 'Layer ' + str(layer) + ' dim 1')
        axes[layer, 0].grid()
        axes[layer, 1].plot(range(self.n_frames), 100*first_sample[:, 0, :]) 
        axes[layer, 1].set(xlabel = 'Frame', ylabel = 'Layer ' + str(layer) + ' dim 2')
        axes[layer, 1].grid()
      else:
        axes[layer, 0].plot(range(self.n_frames), first_sample[:, :, 0]) 
        axes[layer, 0].set(xlabel = 'Frame', ylabel = 'Layer ' + str(layer) + ' dim 1')
        axes[layer, 0].grid()
        axes[layer, 1].plot(range(self.n_frames), first_sample[:, 0, :]) 
        axes[layer, 1].set(xlabel = 'Frame', ylabel = 'Layer ' + str(layer) + ' dim 2')
        axes[layer, 1].grid()
    if show:
      fig.show()
    fig.savefig('./%s/all_layers_vs_frames.png' % (self.model_name))
    plt.close()

  def plot_state_layer(self, x, layer=-1, show=False):
    lat_vars                 = self.model(x)[layer][0]     # sample 0 
    mean_sates_over_channels = np.mean(lat_vars, axis=-1)  # average over the channels
    n_vars                   = lat_vars.shape[1]*lat_vars.shape[2]
    vars_to_plot             = np.zeros((self.n_frames-1, n_vars))
    for t in range(1, self.n_frames): 
      flat_vars = tf.keras.backend.flatten(mean_sates_over_channels[t,:,:]).numpy()
      vars_to_plot[t-1,:] = flat_vars
    plt.figure()
    plt.ylabel('Latent variables')
    plt.xlabel('Frames')
    plt.plot(range(1, self.n_frames), vars_to_plot)
    plt.grid()
    plt.savefig('./%s/layers_vs_frames.png' % (self.model_name))
    if show:
      plt.show()
    plt.close()