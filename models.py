import tensorflow as tf
import matplotlib.pyplot as plt
import imageio


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
my_decoder = tf.keras.Sequential(
    [tf.keras.layers.Flatten(),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Dense(512, activation='relu'   ),
     tf.keras.layers.Dense(  2, activation='softmax')])


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

  def __init__(self, model, reconstructor, decoder, n_frames, model_name):
    super(Wrapper, self).__init__()
    self.model         = model
    self.reconstructor = reconstructor
    self.decoder       = decoder
    self.n_frames      = n_frames
    self.model_name    = model_name

  def call(self, img):
    x      = tf.tile(tf.expand_dims(img, axis=1), (1, self.n_frames, 1, 1, 1))
    states = self.model(x)
    return self.decoder(states[:,0])

  def get_reconstructions(self, x):
    if isinstance(self.model, PredNet):
        return self.model(x)[0]
    else:
      x = tf.cast(x, tf.float32)
      states = self.model(x)
      recs   = []
      for t in range(self.n_frames):
        recs.append(self.reconstructor(states[:,t]))
      return tf.stack(recs, axis=1)
  
  def decode(self, lat_var):
    decs = []
    for n in range(self.n_frames):
      decs.append(self.decoder(lat_var[:, n]))
    return decs

  def compute_rec_loss(self, x, recons):
    weights = [1.0/(n+1) for n in range(self.n_frames-1)]  # first frame cannot be predicted
    if isinstance(self.model, PredNet):                    # nth PredNet output predicts nth frame
      if self.model.t_extrapolate < float('inf'):
        weights = [w if n < self.model.t_extrapolate else 2.0*w for n, w in enumerate(weights)]
      losses = [w*tf.reduce_sum((recons[:, n+1] - x[:, n+1])**2) for n, w in enumerate(weights)]
    else:
      losses = [w*tf.reduce_sum((recons[:, n] - x[:, n+1])**2) for n, w in enumerate(weights)]
    frame_losses = [w*tf.reduce_sum((x[:, n] - x[:, n+1])**2) for n, w in enumerate(weights)]
    return tf.reduce_sum(losses)/tf.reduce_sum(frame_losses)  # new idea!!

  def compute_dec_loss(self, labels, decod):
    criterion = tf.keras.losses.BinaryCrossentropy() # for the moment
    targets = tf.one_hot(labels, 2)
    #weights = [1.0/(n+1) for n in range(self.n_frames-1)]
    weights = [1.0 for n in range (self.n_frames-1)]
    losses = [w*criterion(targets, decod[n+1]) for n, w in enumerate(weights)]
    return tf.reduce_sum(losses) # not normalized

  def train_step(self, x, b, e, opt, labels=None, layer_decod=-1):

    # Train decoding
    if labels is not None:
      with tf.GradientTape() as tape:
        x = tf.cast(x, tf.float32)
        lat_var = self.model(x)[layer_decod]
        decs = self.decode(lat_var)
        dec_loss = self.compute_dec_loss(labels, decs)
        vars_to_train = self.decoder.trainable_variables
      grad = tape.gradient(dec_loss, vars_to_train)
      opt.apply_gradients((zip(grad, vars_to_train)))
      if b == 0:
        self.plot_output(x, self.get_reconstructions(x))
      return dec_loss
      
    # Train reconstruction
    else:
      with tf.GradientTape() as tape:
        x = tf.cast(x, tf.float32)
        recs = self.get_reconstructions(x)
        rec_loss = self.compute_rec_loss(x, recs)
      if isinstance(self.model, PredNet):  # Prednet generates reconstructions itself
        vars_to_train = self.model.trainable_variables
      else:
        vars_to_train = self.model.trainable_variables + self.reconstructor.trainable_variables
      grad = tape.gradient(rec_loss, vars_to_train)
      opt.apply_gradients(zip(grad, vars_to_train))
      if b == 0:
        self.plot_output(x, recs)
      return rec_loss
 
  def plot_output(self, x, r):

    # Plot frames
    r = tf.clip_by_value(r, 0.0, 1.0)
    f = plt.figure(figsize=(int(self.n_frames*(x.shape[3]+3)/32),int(2*(x.shape[2]+3)/32)))
    for t in range(self.n_frames):
      ax1 = f.add_subplot(2, self.n_frames+1, 0*(self.n_frames+1) + t + 1)
      ax2 = f.add_subplot(2, self.n_frames+1, 1*(self.n_frames+1) + t + 1)
      ax1.imshow(tf.squeeze(x[0,t]), cmap='Greys')  # squeeze and cmap only apply to n_channels = 1
      ax2.imshow(tf.squeeze(r[0,t]), cmap='Greys')  # squeeze and cmap only apply to n_channels = 1
    plt.savefig('./%s/latest_input_vs_prediction_epoch.png' % (self.model_name))
    plt.close()

    # Plot gifs
    template  = tf.zeros((10 if i == 3 else x.shape[i] for i in range(len(x.shape))))  # black rectangle
    xr        = tf.concat((x, template, r), axis=3)
    xr_frames = [tf.cast(255*xr[0,t], tf.uint8).numpy() for t in range(self.n_frames)]
    imageio.mimsave('./%s/latest_input_vs_prediction.gif' % (self.model_name), xr_frames, duration=0.1)