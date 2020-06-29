# Dataset funcions
import numpy as np
from numpy.random import RandomState as rng
from skimage.draw import circle, ellipse, rectangle, polygon
# from skimage.morphology import square, rectangle, diamond, disk, octagon, star
from skimage.transform import resize, rotate


# Neil class
class Neil():

	# Initialize object's properties
	def __init__(self, set_type, objects, batch_s, scl, n_frames, c, wn_h, wn_w, grav):

		# Select object static and dynamic properties
		if set_type in ['recons', 'decode']:
			choices    = ['rectangle', 'ellipse', 'vernier'] if set_type == 'recons' else ['vernier']
			x          = rng().uniform( 0,       wn_w,  (1, batch_s))
			y          = rng().uniform( 0,       wn_h,  (1, batch_s))
			vx         = rng().uniform(-5*scl, 5*scl,   (1, batch_s))
			vy         = rng().uniform(-5*scl, 5*scl,   (1, batch_s))
			self.ori   = rng().uniform(0, 2*np.pi,      (1, batch_s))
			self.sizx  = rng().uniform(wn_w/10, wn_w/4, (1, batch_s))  # max: /4
			self.sizy  = rng().uniform(wn_w/10, wn_w/4, (1, batch_s))  # max: /4
			self.colr  = rng().randint(100, 255,        (c, batch_s))
			self.pop_t = rng().randint(0, n_frames//2,  (1, batch_s))
		if set_type == 'sqm':
			choices    = ['vernier']
			flow       = (len(objects)%2 - 0.5)*8*scl**2
			x          = np.ones((1, batch_s))*wn_w//2
			y          = np.ones((1, batch_s))*wn_h//2
			vx         = np.ones((1, batch_s))*flow
			vy         = np.ones((1, batch_s))*0.0
			self.ori   = np.ones((1, batch_s))*0.0
			self.sizx  = np.ones((1, batch_s))*wn_w/5
			self.sizy  = np.ones((1, batch_s))*wn_w/4
			self.colr  = np.ones((c, batch_s), dtype=int)*255
			self.pop_t = np.ones((1, batch_s), dtype=int)*3
		self.shape  = rng().choice( choices, (1, batch_s))
		self.side   = rng().randint(0, 2, (1, batch_s)) if len(objects) == 0 else objects[0].side
		self.side_  = 1*self.side                   # evolving value for sqm (deep copy)
		self.popped = np.array([[False]*batch_s])   # display stimulus or not
		self.sizx[self.shape == 'vernier'] /= 1.5   # verniers look better if not too wide
		self.sizy[self.shape == 'vernier'] *= 2.0   # verniers appear smaller than other shapes
		self.pos    = np.vstack((x,   y))
		self.vel    = np.vstack((vx, vy))
		self.acc    = np.array([[0.00]*batch_s, [grav]*batch_s])
		
		# Generate patches to draw the shapes efficiently
		self.patches = []
		for b in range(batch_s):
			max_s   = int(2*max(self.sizx[0, b], self.sizy[0, b]))
			patch   = np.zeros((max_s, max_s))
			patch_0 = None
			if self.shape[0, b] == 'ellipse':
				center = (patch.shape[0]//2, patch.shape[1]//2)
				radius = (self.sizy[0, b]/2, self.sizx[0, b]/2) 
				rr, cc = ellipse(center[0], center[1], radius[0], radius[1], shape=patch.shape)
				patch[rr, cc] = 255
			elif self.shape[0, b] == 'rectangle':
				start  = (int(max_s - self.sizy[0, b])//2, int(max_s - self.sizx[0, b])//2)
				extent = (int(self.sizy[0, b]), int(self.sizx[0, b]))
				rr, cc = rectangle(start=start, extent=extent, shape=patch.shape)
				patch[rr, cc] = 255
			if self.shape[0, b] == 'vernier':
				patch_0 = np.zeros((max_s, max_s))  # patch with zero offset
				if set_type == 'sqm':
					side    =       self.side[0, b]
					v_siz_w =  1 +  self.sizx[0, b]//4
					v_siz_h =  1 +  self.sizy[0, b]//3
					v_off_w = (1 + (self.sizx[0, b] - v_siz_w)//3)*2
					v_off_h = (1 + (self.sizy[0, b] - v_siz_h)//6)*2 + v_siz_h//2
				else:
					side    = rng().randint(0, 2) if set_type == 'recons' else self.side[0, b]
					v_siz_w = rng().uniform(1 + self.sizx[0, b]//6, 1 + self.sizx[0, b]//2)
					v_siz_h = rng().uniform(1 + self.sizy[0, b]//4, 1 + self.sizy[0, b]//2)
					v_off_w = rng().uniform(1,              1 + (self.sizx[0, b] - v_siz_w)//2)*2
					v_off_h = rng().uniform(1 + v_siz_h//2, 1 + (self.sizy[0, b] - v_siz_h)//2)*2
					if len(objects) > 0 and set_type == 'decode':
						v_off_w = 0.0  # only one vernier (the first in the list) has an offset in decode mode
				start1     = (int((max_s - v_off_h - v_siz_h)//2), int((max_s - v_off_w - v_siz_w)//2))
				start2     = (int((max_s + v_off_h - v_siz_h)//2), int((max_s + v_off_w - v_siz_w)//2))
				start01    = (int((max_s - v_off_h - v_siz_h)//2), int((max_s - 0       - v_siz_w)//2))
				start02    = (int((max_s + v_off_h - v_siz_h)//2), int((max_s + 0       - v_siz_w)//2))
				extent     = (int(v_siz_h), int(v_siz_w))
				rr1,  cc1  = rectangle(start=start1,  extent=extent, shape=patch.shape)
				rr2,  cc2  = rectangle(start=start2,  extent=extent, shape=patch.shape)
				rr01, cc01 = rectangle(start=start01, extent=extent, shape=patch.shape)
				rr02, cc02 = rectangle(start=start02, extent=extent, shape=patch.shape)
				patch[  rr1,  cc1 ] = 255
				patch[  rr2,  cc2 ] = 255
				patch_0[rr01, cc01] = 255
				patch_0[rr02, cc02] = 255
			patch  = rotate(patch, self.ori[0, b]).astype(int)
			to_add = [patch, np.fliplr(patch), patch_0]
			self.patches.append(to_add)

	# Compute what must be updated between the frames
	def compute_changes(self, t, batch_s, objects, set_type, cond):
		
		# Visible objects appear
		self.popped[t >= self.pop_t] = True

		# SQM related changes
		if set_type == 'sqm':
			condition = cond[:-1]
			change_t  = int(cond[-1])
			for b in range(batch_s):
				if t == self.pop_t[0, b]:
					self.side_[:, b] = self.side[:, b]                 # seed offset
				elif change_t > 0 and t == self.pop_t[0, b] + change_t:
					if condition == 'V-AV':
						objects[-1].side_[:, b] = 1 - self.side[:, b]  # opposite offset	
					if condition == 'V-PV':
						objects[-1].side_[:, b] = self.side[:, b]      # same offset
				else:
					self.side_[:, b] = 2                               # no offset

	# Draw the object (square patch)
	def draw(self, wn, batch_s):
		# print(self.side_[0, 0])
		# print(self.side[ 0, 0])
		for b in range(batch_s):
			if self.popped[:, b]:
				patch  = self.patches[b][self.side_[0, b]]/255
				start  = [self.pos[1, b] - patch.shape[0]//2, self.pos[0, b] - patch.shape[1]//2]
				rr, cc = rectangle(start=start, extent=patch.shape, shape=wn.shape[1:3])
				rr     = rr.astype(int)
				cc     = cc.astype(int)
				pat_rr = (rr - self.pos[1, b] - patch.shape[0]/2).astype(int)
				pat_cc = (cc - self.pos[0, b] - patch.shape[1]/2).astype(int)
				bckgrd = wn[b, rr, cc, :]
				for i, color in enumerate(self.colr[:, b]):
					col_patch = color*patch[pat_rr, pat_cc] - bckgrd[:,:,i]
					wn[b, rr, cc, i] += col_patch.clip(0, 255).astype(np.uint8)

	# Update objects position and velocity (of visible objects)
	def update_states(self, batch_s, friction):
		self.vel[:, self.popped[0]] += self.acc[:, self.popped[0]] - self.vel[:, self.popped[0]]*friction
		self.pos[:, self.popped[0]] += self.vel[:, self.popped[0]]


# Class to generate batches of bouncing balls
class BatchMaker():

	# Initiates all values unchanged from batch to batch
	def __init__(self, set_type, n_objects, batch_s, n_frames, im_dims, condition='V'):
		self.set_type   = set_type
		self.Object     = Neil  # for now
		self.n_objects  = n_objects
		self.n_max_occl = 0
		self.condition  = condition if condition != 'V' else 'V0'  # coding detail
		self.batch_s    = batch_s
		self.n_frames   = n_frames
		self.n_chans    = im_dims[-1]
		self.scale      = max(im_dims[0], im_dims[1])/64
		self.wn_h       = int(im_dims[0]*self.scale)
		self.wn_w       = int(im_dims[1]*self.scale)
		self.gravity    = 0.0
		self.friction   = 0.0
	
	# Initialize batch, objects (size, position, velocities, etc.) and background
	def init_batch(self):

		# Background window and objects inside it
		self.batch    = []
		self.objects  = []
		self.window   = 127*np.ones((self.batch_s, self.wn_h, self.wn_w, self.n_chans), dtype=int)
		for _ in range(self.n_objects):
			self.objects.append(self.Object(self.set_type, self.objects, self.batch_s, self.scale,
							    self.n_frames, self.n_chans, self.wn_h, self.wn_w, self.gravity))
		self.bg_color = rng().randint(0, 80, (self.batch_s, self.n_chans))  # if set_type == 'recons' else 40*np.ones((self.batch_s, self.n_chans))
		for b in range(self.batch_s):
			for c in range(self.n_chans):
				self.window[b, :, :, c] = self.bg_color[b, c]

		# Occluding walls in the frontground
		n_occl        = rng().randint(0, self.n_max_occl+1, (self.batch_s)) if self.set_type == 'recons' else [0]*self.batch_s
		self.frnt_grd = np.zeros(self.window.shape, dtype=bool)
		for b in range(self.batch_s):
			for _ in range(n_occl[b]):
				if rng().rand() > 0.5:
					pos    = rng().randint(0, self.wn_h)
					height = rng().randint(2, self.wn_h//10)
					self.frnt_grd[b, max(0, pos-height):min(self.wn_h, pos+height), :, :] = True
				else:
					pos    = rng().randint(0, self.wn_w)
					height = rng().randint(2, self.wn_w//10)
					self.frnt_grd[b, :, max(0, pos-height):min(self.wn_w, pos+height), :] = True

	# Batch making function (generating batch_s dynamic sequences)
	def generate_batch(self):
		self.init_batch()
		for t in range(self.n_frames):
	
			# Compute and draw moving objects
			frame = self.window*1
			for i, obj in enumerate(self.objects):
				obj.compute_changes(t, self.batch_s, self.objects, self.set_type, self.condition)
			for obj in self.objects:
				obj.draw(frame, self.batch_s)
			for obj in self.objects:
				obj.update_states(self.batch_s, self.friction)

			# Add noise and black frontground walls
			frame[self.frnt_grd] = 0.0
			self.batch.append(frame.clip(0, 255).astype(np.uint8))

		# Return batch (and labels)
		if self.set_type == 'recons':
			return self.batch, None                     # list of n_frames numpy arrays of dims [batch, h, w, channels]
		else:
			return self.batch, self.objects[0].side[0]  # the label is always the vernier(s) offset in the first frame


# Show example of reconstruction batch
if __name__ == '__main__':

  import pyglet   # conda install -c conda-forge pyglet
  import imageio  # conda install -c conda-forge imageio
  import os
  set_type     = 'sqm'    # 'recons', 'decode' or 'sqm'
  condition    = 'V-PV3'  # 'V', 'V-PVn' or 'V-AVn', n > 0
  n_objects    = 2
  n_frames     = 13
  scale        = 2
  batch_s      = 4
  n_channels   = 3
  batch_maker  = BatchMaker(set_type, n_objects, batch_s, n_frames, (64*scale, 64*scale, 1), condition)
  if set_type == 'recons':
  	batch_frames = batch_maker.generate_batch()
  else:
  	batch_frames, _ = batch_maker.generate_batch()
  gif_name        = 'test_output.gif'
  display_frames  = []
  for t in range(n_frames):
  	display_frames.append(np.hstack([batch_frames[t][b] for b in range(batch_s)]))
  imageio.mimsave(gif_name, display_frames, duration=0.1)
  anim   = pyglet.resource.animation(gif_name)
  sprite = pyglet.sprite.Sprite(anim)
  window = pyglet.window.Window(width=sprite.width, height=sprite.height)
  window.set_location(600, 300)
  @window.event
  def on_draw():
  	window.clear()
  	sprite.draw()
  pyglet.app.run()
  os.remove(gif_name)