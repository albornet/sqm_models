# Dataset funcions
import numpy as np
from numpy.random import RandomState as rng
from skimage.draw import circle, ellipse, rectangle, polygon
from skimage.transform import rotate


# Ball class
class Ball():

	# Function to create the ball
	def __init__(self, set_type, balls, batch_s, scale, n_chans, wn_h, wn_w, wall_d, grav):

		# Select id, color, rebound fact, size, mass and gravity
		self.colr  = rng().randint(    100,    255, (n_chans, batch_s))
		self.fact  = rng().uniform(    0.8,    0.9, (1,       batch_s))
		self.size  = rng().uniform(wn_w/14, wn_w/7, (1,       batch_s))
		self.mass  = self.size**3

		# Booleans initialization
		self.tch_G = np.zeros((1, batch_s), dtype=bool)  # touches ground
		self.tch_R = np.zeros((1, batch_s), dtype=bool)  # touches right wall
		self.tch_L = np.zeros((1, batch_s), dtype=bool)  # touches left  wall

		# Select random initial position, velocity and acceleration
		w        = wall_d
		wn_d     = max(wn_w, wn_h)
		s        = max(self.size)
		x        = rng().uniform(s+w,       wn_d-s-w, (1, batch_s))
		y        = rng().uniform(  0,       wn_d-s-w, (1, batch_s))
		vx       = rng().uniform(-15*scale, 15*scale, (1, batch_s))
		vy       = rng().uniform( -5*scale,  0*scale, (1, batch_s))
		self.pos = np.vstack((x,   y))
		self.vel = np.vstack((vx, vy))
		self.acc = np.array([[0.00]*batch_s, [grav]*batch_s])

		# Check for overlap with other balls
		if len(balls) > 1:
			bad = np.array([True]*batch_s)
			while any(bad):
				bad = np.array([False]*batch_s)
				for ball in balls:
					d = np.hypot(*(self.pos-ball.pos))
					bad[(d < self.size + ball.size)[0]] = True

				# Try new sizes and positions if the ball overlaps with another one
				self.size[:, bad] = rng().uniform(wn_d/12, wn_d/6, (1, bad.sum()))
				self.mass[:, bad] = self.size[:, bad]**3
				s                 = max(self.size[:, bad])
				x                 = rng().uniform(s+w, wn_w-s-w, (1, bad.sum()))
				y                 = rng().uniform(  0, wn_h-s-w, (1, bad.sum()))
				self.pos[:, bad]  = np.vstack((x, y))

	# Take care of ball chocs
	def compute_changes(self, balls, self_idx, wn_h, wn_w, wall_d):

		# Check balls that move towards each other
		for i in range(self_idx+1, len(balls)):
			ball = balls[i]
			if ball is not self:
				r12 = self.pos - ball.pos
				d   = np.hypot(*r12)
				bad = (d < self.size + ball.size)[0]
				if bad.sum() > 0:

					# Balls hit each other
					dT     = d[bad]
					rT     = r12[:, bad]
					v1, v2 = self.vel[ :, bad], ball.vel[ :, bad]
					m1, m2 = self.mass[:, bad], ball.mass[:, bad]
					vT     = ((v1-v2)*rT).sum(axis=0)
					fT     = 1.0 - (1.0-self.fact[:, bad]) - (1.0-ball.fact[:, bad])
					u1     = (2*m2/(m1+m2)*vT/(dT**2)*rT)
					u2     = (2*m1/(m1+m2)*vT/(dT**2)*rT)
					self.vel[:, bad]  = (self.vel[:, bad] - u1)*fT
					ball.vel[:, bad]  = (ball.vel[:, bad] + u2)*fT

		# Check balls that hit walls
		self.tch_G = (self.pos[1] + 1 > -self.size-wall_d+wn_h)  # touches ground
		self.tch_R = (self.pos[0] + 1 > -self.size-wall_d+wn_w)  # touches right wall
		self.tch_L = (self.pos[0] - 1 <  self.size+wall_d     )  # touches left wall

		# Check balls that touch each other
		for i in range(0, len(balls)):
			ball = balls[i]
			if ball is not self:
				r12  = self.pos - ball.pos
				d    = np.hypot(*r12)
				bad  = (d     < self.size + ball.size)[0]
				badG = (d - 0 < self.size + ball.size)[0]
				bad  = np.logical_and(bad, r12[1] > 0)
				badG = np.logical_and(np.logical_and(badG, r12[1] > 0), self.tch_G[0])
				if bad.sum() > 0:

					# Correct ball positions
					dT = d[bad]
					rT = r12[:, bad]
					cT = rT/dT*(self.size[:, bad] + ball.size[:, bad] - dT)
					self.pos[:, bad ] += cT/2
					ball.pos[:, bad ] -= cT/2

					# Correct how gravity applies to balls on top
					aT = self.acc[:, badG]
					gT = r12[:, badG]/d[badG]        # unit vector
					gT = np.vstack((-gT[1], gT[0]))  # perp unit vector
					gT = (gT*aT).sum(axis=0)*gT      # perp vector scalar normed
					ball.acc[:, badG] = gT

		# Re-check and correct balls that hit walls
		self.tch_G = (self.pos[1] > -self.size-wall_d+wn_h)  # touches ground
		self.tch_R = (self.pos[0] > -self.size-wall_d+wn_w)  # touches right wall
		self.tch_L = (self.pos[0] <  self.size+wall_d     )  # touches left wall
		self.vel[1, self.tch_G[0]] *= -self.fact[self.tch_G]
		self.vel[0, self.tch_R[0]] *= -self.fact[self.tch_R]
		self.vel[0, self.tch_L[0]] *= -self.fact[self.tch_L]
		self.pos[1, self.tch_G[0]]  = -self.size[self.tch_G] - wall_d + wn_h
		self.pos[0, self.tch_R[0]]  = -self.size[self.tch_R] - wall_d + wn_w
		self.pos[0, self.tch_L[0]]  =  self.size[self.tch_L] + wall_d

	# Draw the ball (probably this can be done much more efficiently)
	def draw(self, wn, batch_s):
		for b in range(batch_s):  # (row, col) is (y, x)
			rc, cc = circle(self.pos[1, b], self.pos[0, b], self.size[0, b], shape=wn.shape[1:3])
			for i, color in enumerate(self.colr[:, b]):
				wn[b, rc, cc, i] = color
			# p0 = self.pos[:, b].astype(int)
			# p1 = (self.pos[:, b] + 5*self.acc[:, b]).astype(int)
			# rc, cc = line(p0[1], p0[0], p1[1], p1[0])
			# wn[b, rc, cc, :] = 0

	# Update position and velocity, reset acceleration
	def update_states(self, batch_s, friction, gravity):
		self.vel += self.acc - self.vel*friction
		self.pos += self.vel
		self.acc  = np.array([[0.00]*batch_s, [gravity]*batch_s])


# Neil class
class Neil():

	# Initialize object's properties
	def __init__(self, set_type, objects, batch_s, scale, n_chans, wn_h, wn_w, wall_d, grav):

		# Select id, color, sizes, orientations, etc.
		if set_type == 'recons':
			#choices    = ['rectangle', 'ellipse', 'vernier']
			choices    = ['vernier']
			self.ori   = rng().uniform(0, 2*np.pi, (1, batch_s))
		if set_type == 'decode':
			choices    = ['vernier']
			self.vside = rng().randint(0, 2,       (1, batch_s)) if objects == [] else objects[0].vside
			self.ori   = rng().uniform(0, np.pi/2, (1, batch_s))
		self.shape = rng().choice(choices,          (1,       batch_s))
		self.colr  = rng().randint(100, 255,        (n_chans, batch_s))
		self.sizx  = rng().uniform(wn_w/10, wn_w/4, (1,       batch_s))
		self.sizy  = rng().uniform(wn_w/10, wn_w/4, (1,       batch_s))
		self.sizx[self.shape == 'vernier'] /= 1.5  # verniers look better if not too wide
		self.sizy[self.shape == 'vernier'] *= 2.0  # verniers appear smaller than other shapes
		
		# Select random initial position, velocity and acceleration
		x        = rng().uniform( 2*wn_w/6, 4*wn_w/6, (1, batch_s))
		y        = rng().uniform( 2*wn_h/6, 4*wn_h/6, (1, batch_s))
		vx       = rng().uniform(-5*scale,  5*scale,  (1, batch_s))
		vy       = rng().uniform(-5*scale,  5*scale,  (1, batch_s))
		self.pos = np.vstack((x,   y))
		self.vel = np.vstack((vx, vy))
		self.acc = np.array([[0.00]*batch_s, [grav]*batch_s])
		self.patches = self.generate_patches()
	
	# Generate patches to draw the shapes efficiently
	def generate_patches(self):
		patches = []
		for b in range(batch_s):
			max_s = int(2*max(self.sizx[0, b], self.sizy[0, b]))
			patch = np.zeros((max_s, max_s))
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
			# if self.shape == 'triangle':
			# 	rr, cc = polygon(...)
			
			if self.shape[0, b] == 'vernier':
				vside    = rng().randint(0, 2) if set_type == 'recons' else self.vside[0, b] 
				v_siz_w  = rng().uniform(1 + self.sizx[0, b]//6, 1 + self.sizx[0, b]//2) 
				v_siz_h  = rng().uniform(1 + self.sizy[0, b]//4, 1 + self.sizy[0, b]//2) 
				v_off_w  = rng().uniform(1,              1 + (self.sizx[0, b] - v_siz_w)//2)*2 
				v_off_h  = rng().uniform(1 + v_siz_h//2, 1 + (self.sizy[0, b] - v_siz_h)//2)*2 
				start1   = (int((max_s - v_off_h - v_siz_h)//2), int((max_s - v_off_w - v_siz_w)//2))
				start2   = (int((max_s + v_off_h - v_siz_h)//2), int((max_s + v_off_w - v_siz_w)//2))
				extent   = (int(v_siz_h), int(v_siz_w))
				rr1, cc1 = rectangle(start=start1, extent=extent, shape=patch.shape)
				rr2, cc2 = rectangle(start=start2, extent=extent, shape=patch.shape)
				patch[rr1, cc1] = 255
				patch[rr2, cc2] = 255
				if vside:  # 0 is R vernier and 1 is L vernier 
					patch = np.fliplr(patch) 
			patches.append(rotate(patch, self.ori[0, b]).astype(int))
		return patches
		

	# Compute what must be updated between the frames
	def compute_changes(self, objects, self_idx, wn_h, wn_w, wall_d, t):
		pass

	# Draw the object (square patch)
	def draw(self, wn, batch_s):
		for b in range(batch_s):
			patch  = self.patches[b]/255
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

	# Update objects position and velocity
	def update_states(self, batch_s, friction, gravity):
		self.vel += self.acc - self.vel*friction
		self.pos += self.vel


class SQM(Neil):
	# Initialize object's properties
	def __init__(self, set_type, objects, batch_s, scale, n_chans, wn_h, wn_w, wall_d, grav, condition, side):
		super().__init__(set_type, objects, batch_s, scale, n_chans, wn_h, wn_w, wall_d, grav)
	
		self.ori   = np.zeros((1, batch_s)) 
		self.colr = 150*np.ones((n_chans, batch_s))
		self.sizx  = wn_w/10*np.ones((1,       batch_s))
		self.sizy  = wn_w/3*np.ones((1,       batch_s))

		self.condition = condition
		self.side = side # 0 for the right line and 1 for the left one
		self.vside = rng().randint(0, 2,       (1, batch_s))
		if self.condition == 'V' or self.condition == 'V-AV' or self.condition == 'V-PV':
			if self.side == 0: # Flip juste one of the 2 lines for the vernier in the first frame
				self.generate_patches(True) 
			else:
				self.generate_patches(False)
		else: 
			self.generate_patches(False)
		
		x = wn_w/2*np.ones((1, batch_s)) 
		y = wn_h/2*np.ones((1, batch_s))
		self.pos = np.vstack((x,   y))
		# Set the velocity sign in function of the relative position of the line
		if self.side == 0:
			vx = 1*scale*np.ones((1, batch_s)) # for the moment (to adapt with the number of frames or code something to return at the center of the window)
		else:
			vx = -1*scale*np.ones((1, batch_s))
		vy = np.zeros((1, batch_s))
		self.vel = np.vstack((vx, vy))
		self.acc = np.array([[0.00]*batch_s, [0.00]*batch_s]) 
	
	# Generate patches to draw the shapes efficiently
	def generate_patches(self, offset = False):
		self.patches = []
		for b in range(batch_s):
			max_s = int(2*max(self.sizx[0, b], self.sizy[0, b]))
			patch = np.zeros((max_s, max_s))
			v_siz_w  = self.sizx[0, b]//4 
			v_siz_h  = self.sizy[0, b]
			# Generate the horizontal offset if the offset condition is True
			if offset:
				v_off_w  = 1 + (self.sizx[0, b] - v_siz_w)//2 
				v_off_h  = self.sizy[0, b] 
			else:
				v_off_w = 0
				v_off_h =  self.sizy[0, b] 
			start1   = (int((max_s - v_off_h - v_siz_h)//2), int((max_s - v_off_w - v_siz_w)//2)) 
			start2   = (int((max_s + v_off_h - v_siz_h)//2), int((max_s + v_off_w - v_siz_w)//2))
			extent   = (int(v_siz_h), int(v_siz_w))
			rr1, cc1 = rectangle(start=start1, extent=extent, shape=patch.shape)
			rr2, cc2 = rectangle(start=start2, extent=extent, shape=patch.shape)
			patch[rr1, cc1] = 255
			patch[rr2, cc2] = 255
			if offset and self.vside[0, b]: # 0 is R vernier and 1 is L vernier 
				patch = np.fliplr(patch) 
			self.patches.append(rotate(patch, self.ori[0, b]).astype(int))
	
	# Invert the offset direction (for the vernier-antivernier condition V-AV)
	def inv_offset(self):
		for i, vside in enumerate(self.vside[0]):
			if vside == 1:
				self.vside[0, i] = 0
			else:
				self.vside[0, i] = 1

	# Compute what must be updated between the frames
	def compute_changes(self, objects, self_idx, wn_h, wn_w, wall_d, t):
		# After the first frame, reset the configuration to no visible offset anymore
		if t == 1:
			self.generate_patches(False)
		# At the 5'th frame, generate an offset in the V-AV and V-PV conditions
		if t == 4:
			if self.condition == 'V-AV':
				if self.side == 0:
					self.inv_offset()
					self.generate_patches(True)	
			if self.condition == 'V-PV':
				if self.side == 0:
					self.generate_patches(True)
		# After the offset of frame 5, reset the configuration to no visible offset again
		if t == 5:
			self.generate_patches(False)
		pass


# Class to generate batches of bouncing balls
class BatchMaker():

	# Initiates all values unchanged from batch to batch
	def __init__(self, set_type, object_type, n_objects, batch_s, n_frames, im_dims, condition = None):
		object_dict = {
		  'ball': {'generator': Ball, 'gravity': 1.00, 'wall_d': 4, 'friction': 0.01},
		  'neil': {'generator': Neil, 'gravity': 0.00, 'wall_d': 0, 'friction': 0.00},
		  'sqm': {'generator': SQM, 'gravity': 0.00, 'wall_d': 0, 'friction': 0.00}}
		object_descr   = object_dict[object_type]
		self.Object    = object_descr['generator']
		self.n_objects = n_objects
		self.condition = condition
		self.batch_s   = batch_s
		self.n_frames  = n_frames
		self.n_chans   = im_dims[-1]
		self.scale     = max(im_dims[0], im_dims[1])/64
		self.wn_h      = int(im_dims[0]*self.scale)
		self.wn_w      = int(im_dims[1]*self.scale)
		self.wall_d    = int(object_descr['wall_d']*self.scale)
		self.gravity   = object_descr['gravity']*self.scale
		self.friction  = object_descr['friction']
		self.set_type  = set_type
	
	# Initialize batch, objects (size, position, velocities, etc.) and background
	def init_batch(self):
		self.batch   = []
		self.objects = []
		self.window  = 127*np.ones((self.batch_s, self.wn_h, self.wn_w, self.n_chans), dtype=np.uint8)
		
		for _ in range(self.n_objects):
			if (self.Object == SQM):
				self.objects.append(self.Object(  self.set_type, self.objects, self.batch_s, self.scale,
			                    self.n_chans, self.wn_h,     self.wn_w,    self.wall_d,  self.gravity, self.condition, _))
			else:
				self.objects.append(self.Object(  self.set_type, self.objects, self.batch_s, self.scale,
			                    self.n_chans, self.wn_h,     self.wn_w,    self.wall_d,  self.gravity, self.condition))
		self.bg_color = rng().randint(0, 80, (self.batch_s, self.n_chans))
		for b in range(self.batch_s):
			for c in range(self.n_chans):
				self.window[b, :self.wn_h-self.wall_d, self.wall_d:self.wn_w-self.wall_d, c] = self.bg_color[b, c]

	# Batch making function (generating batch_s dynamic sequences)
	def generate_batch(self):
		self.init_batch()
		for t in range(self.n_frames):
			frame = self.window*1
			for i, obj in enumerate(self.objects):
				if isinstance(obj, SQM):
					obj.compute_changes(self.objects, i, self.wn_h, self.wn_w, self.wall_d, t)
				else:
					obj.compute_changes(self.objects, i, self.wn_h, self.wn_w, self.wall_d)
			for obj in self.objects:
				obj.draw(frame, self.batch_s)
			for obj in self.objects:
				obj.update_states(self.batch_s, self.friction, self.gravity)
			self.batch.append(frame)
		return self.batch  # list of n_frames numpy arrays of dims [batch, h, w, channels]


# Show example of reconstruction batch
if __name__ == '__main__':

  import pyglet   # conda install -c conda-forge pyglet
  import imageio  # conda install -c conda-forge imageio
  import os
  object_type  = 'sqm'
  set_type     = 'decode'
  condition    = 'V-AV'
  n_objects    = 2
  n_frames     = 10
  scale        = 2
  batch_s      = 4
  n_channels   = 3
  batch_maker  = BatchMaker(set_type, object_type, n_objects, batch_s, n_frames, (64*scale, 64*scale, n_channels), condition)

  gif_name        = 'test_output.gif'
  batch_frames = batch_maker.generate_batch()
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