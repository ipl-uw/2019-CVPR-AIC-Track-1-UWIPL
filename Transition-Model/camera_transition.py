'''
Tsung-Wei Huang
AI City 2019
'''


import xml.etree.ElementTree as ET
import csv

import cv2 # for visualization
import math # for inf

CHECK_TRACK_ORDER = False

'''
flag definition:
	1) IS in transition time
	0) NOT in transition time
	-1) transition time for camera pair not defined
	-2) transition time for zonelist pair not defined
	-3) time information for track to calculate transition time not exist
'''


def zonelists_to_str(zonelists):
	zss = [zonelist_to_str(zonelist) for zonelist in zonelists]
	return ';'.join(zss)

def zonelist_to_str(zonelist):
	zs = [str(z) for z in zonelist]
	return '-'.join(zs)

def parse_zonelist(st):
    st = st.split('-')
    st = tuple([int(s) for s in st])
    return st

def parse_zonelists(st):
    zonelists = st.split(';')
    zonelists = [parse_zonelist(zonelist) for zonelist in zonelists]
    return zonelists

def zone_box_overlap(zone, box, verbose=False):
	# zone, box: (x, y, w, h)
	dx = min(zone[0]+zone[2], box[0]+box[2]) - max(zone[0], box[0])
	if dx <= 0:
		return 0
	dy = min(zone[1]+zone[3], box[1]+box[3]) - max(zone[1], box[1])
	if dy <= 0:
		return 0
	return (dx*dy)/float(box[2]*box[3])

def zone_boxes_overlap(zone, boxes, verbose=False):
	# zone: (x, y, w, h)
	# boxes: [(x, y, w, h)]
	overlaps = []
	for box in boxes:
		overlaps.append(zone_box_overlap(zone, box, verbose))
	if verbose and False:
		print('zone:' + str(zone))
		print('boxes:' + str(boxes))
		print('overlaps:' + str(overlaps))
	overlap_max = max(overlaps)
	if overlap_max > 0:
		ids = [i for i, overlap in enumerate(overlaps) if overlap == overlap_max]
		return ids[int(len(ids)/2)], overlap_max
	else:
		return -1, 0

def valid_order(src, dst):
	pos = -1
	for x in src:
		if x in dst:
			p = dst.index(x)
			if p < pos:
				return False
			else: # non-decreasing is ok
				pos = p
	return True

def zonelist_dist(zonelist, zs, overlaps):
	if not valid_order(zonelist, zs) or not valid_order(zs, zonelist):
		return len(zonelist)+len(zs)
	dist = 0
	for z in zonelist:
		if z not in zs:
			dist += 1
	for i, z in enumerate(zs):
		if z in zonelist:
			dist += 1 - overlaps[i]
		else:
			dist += overlaps[i]
	return dist

class Camera:
	def __init__(self, camId):
		self.id = camId
		self.zones = {}
		self.zonelists = []
		self.timestamps = [] # sorted list [(t, dt)]

	def add_zone(self, name, x, y, w, h, verbose=False):
		assert name not in self.zones
		self.zones[name] = (x, y, w, h)

	def read_zone_xml(self, filename, verbose=False):
		self.zones = {}
		tree = ET.parse(filename)
		root = tree.getroot()
		for obj in root.iter('object'):
			name = int(obj.find('name').text)
			bndbox = obj.find('bndbox')
			xmin = int(bndbox.find('xmin').text)
			ymin = int(bndbox.find('ymin').text)
			xmax = int(bndbox.find('xmax').text)
			ymax = int(bndbox.find('ymax').text)
			# convert to (x, y, w, h)
			self.zones[name] = (xmin, ymin, xmax - xmin, ymax - ymin)
		if verbose:
			print(self.id)
			for name in self.zones:
				print(self.zones[name])

	def add_zonelist(self, zonelist):
		for zone in zonelist:
			assert zone in self.zones
		if zonelist not in self.zonelists:
			self.zonelists.append(zonelist)

	def add_timestamp(self, t, dt):
		self.timestamps.append((t, dt))
		self.timestamps.sort()

	def correct_time(self, t):
		# calculate correct timestamp
		if len(self.timestamps) == 0:
			return t
		if len(self.timestamps) == 1:
			return t + self.timestamps[0][1]
		if t < self.timestamps[0][0]:
			return t + self.timestamps[0][1]
		if t >= self.timestamps[-1][0]:
			return t + self.timestamps[-1][1]
		i_left = 0
		for i in range(len(self.timestamps) - 1):
			if self.timestamps[i+1][0] > t:
				i_left = i
				break
			else:
				continue
		t_left, dt_left = self.timestamps[i_left]
		t_right, dt_right = self.timestamps[i_left+1]
		assert t_left <= t < t_right
		dt = (dt_left * (t_right - t) + dt_right * (t - t_left)) / (t_right - t_left)
		return int(t + dt) ### always int

	def match_zonelist_track(self, track, verbose=False):
		# track: [(camId, carId, t, x, y, w, h)]
		#if verbose:
		#	print('camId: %d'%self.id)
		carId = -1
		t = -math.inf
		for tr in track:
			assert tr[0] == self.id
			if carId == -1:
				carId = tr[1]
			else:
				assert tr[1] == carId
			if CHECK_TRACK_ORDER:
				assert tr[2] > t # track must be strictly increasing in time
				t = tr[2]

		if verbose:
			print('camId: %d'%self.id)

		boxes = [tr[3:7] for tr in track]
		z_it_overlap = []
		for z, zone in self.zones.items():
			#if verbose:
			#	print('z: %d'%z)
			it, overlap = zone_boxes_overlap(zone, boxes, verbose)
			if overlap > 0:
				z_it_overlap.append((z, it, overlap))
		# TODO: if len(z_it_overlap) == 0: # no overlap with any zone
		if len(z_it_overlap) == 0:
			# probably bad track, just skip it
			if verbose:
				print('no overlap with any zone')
			return (), []
		z_it_overlap.sort(key=lambda x: x[1])
		if verbose:
			#print('carId: %d'%carId)
			#print('camId: %d'%self.id)
			print('start time: %d'%track[0][2])
			print('z_it_overlap:')
			print(z_it_overlap)
		zs = [x[0] for x in z_it_overlap]
		its = [x[1] for x in z_it_overlap]
		overlaps = [x[2] for x in z_it_overlap]
		dists = []
		for zonelist in self.zonelists:
			dist = zonelist_dist(zonelist, zs, overlaps)
			dists.append(dist)
		if verbose:
			print('zonelists:')
			print(self.zonelists)
			print('dists:')
			print(dists)
		dist_min = min(dists)
		# TODO: if there are more than one minimum
		idxs = [idx for idx in range(len(dists)) if dists[idx] == dist_min]
		if len(idxs) > 1:
			# probably bad track, just skip it
			if verbose:
				print('more than one minimum zonelist')
			return (), []

		zonelist_min = self.zonelists[dists.index(dist_min)]
		its_min = []
		for z in zonelist_min:
			if z in zs:
				it = its[zs.index(z)]
			else:
				it = -1
			its_min.append(it)
		#ts_min = [(track[it][2] if it >= 0 else -math.inf) for it in its_min]
		ts_min = [(self.correct_time(track[it][2]) if it >= 0 else -1) for it in its_min]
		if verbose:
			print('zonelist_min:')
			print(zonelist_min)
			print('ts_min:')
			print(ts_min)

		return zonelist_min, ts_min

	def visualize_zones(self, input_filename, output_filename, color=(0,255,0), thickness=4, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=4, text_color=(0,0,255), text_thickness=4):
		img = cv2.imread(input_filename)
		for name, (x, y, w, h) in self.zones.items():
			cv2.rectangle(img, (x, y), (x+w, y+h), color, thickness)
		#for zonelist in self.zonelists:
		#	for i, z in enumerate(zonelist):
		for name, (x, y, w, h) in self.zones.items():
			cv2.putText(img, str(name), (int(x+w/2), int(y+h/2)), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

		text = 'c%03d' % (self.id)
		textsize = cv2.getTextSize(text, font, font_scale, text_thickness)[0]
		# get coords based on boundary
		textX = (img.shape[1] - textsize[0])
		textY = (textsize[1])
		# add text centered on image
		cv2.putText(img, text, (textX, textY), font, font_scale, (255, 0, 0), text_thickness)
		cv2.imwrite(output_filename, img)



class Transition:
	def __init__(self, camId_src, camId_dst):
		self.camId_src = camId_src
		self.camId_dst = camId_dst
		self.zonelists_src = []
		self.zonelists_dst = []
		self.z_src = -1
		self.z_dst = -1
		self.dt_min = -1
		self.dt_max = -1
		self.distribution = [] # [(t_src, dt)]

	# obsolete...
	#def __lt__(self, other):
	#	return (self.zonelists_src, self.zonelists_dst) < (other.zonelists_src, other.zonelists_dst)

	def set_zonelists(self, zonelists_src, zonelists_dst, z_src, z_dst, dt_min, dt_max):
		for zonelist in zonelists_src:
			assert z_src in zonelist
		for zonelist in zonelists_dst:
			assert z_dst in zonelist
		self.zonelists_src = zonelists_src
		self.zonelists_dst = zonelists_dst
		self.z_src = z_src
		self.z_dst = z_dst
		self.dt_min = dt_min
		self.dt_max = dt_max

	def update_transition(self, zonelist_src, zonelist_dst, ts_src, ts_dst, verbose=False):
		if zonelist_src not in self.zonelists_src or zonelist_dst not in self.zonelists_dst:
			return -2
		if verbose:
			print('(zonelist_src, zonelist_dst): (%s, %s)'%(zonelist_to_str(zonelist_src), zonelist_to_str(zonelist_dst)))
		t_src = ts_src[zonelist_src.index(self.z_src)]
		t_dst = ts_dst[zonelist_dst.index(self.z_dst)]
		if verbose:
			print('(t_src, t_dst): (%d, %d)'%(t_src, t_dst))
		if t_src < 0 or t_dst < 0:
			return -3
		dt = t_dst - t_src
		if verbose:
			print('dt: %d'%dt)
		if verbose:
			print('old (dt_min, dt_max): (%d, %d)'%(self.dt_min, self.dt_max))
		self.dt_min = min(self.dt_min, dt)
		self.dt_max = max(self.dt_max, dt+1)
		if verbose:
			print('new (dt_min, dt_max): (%d, %d)'%(self.dt_min, self.dt_max))
		self.distribution.append((t_src, dt))
		return 1

	def in_transition(self, zonelist_src, zonelist_dst, ts_src, ts_dst, w_min=0, w_max=0, verbose=False):
		if zonelist_src not in self.zonelists_src or zonelist_dst not in self.zonelists_dst:
			return -2
		if verbose:
			print('(zonelist_src, zonelist_dst): (%s, %s)'%(zonelist_to_str(zonelist_src), zonelist_to_str(zonelist_dst)))
		t_src = ts_src[zonelist_src.index(self.z_src)]
		t_dst = ts_dst[zonelist_dst.index(self.z_dst)]
		if verbose:
			print('(t_src, t_dst): (%d, %d)'%(t_src, t_dst))
		if t_src < 0 or t_dst < 0:
			return -3
		dt = t_dst - t_src
		if verbose:
			print('dt: %d'%dt)
		if verbose:
			print('(dt_min, dt_max): (%d, %d)'%(self.dt_min, self.dt_max))
			print('(w_min, w_max): (%d, %d)'%(w_min, w_max))
		if dt >= self.dt_min - w_min and dt < self.dt_max + w_max:
			if verbose:
				print('in range')
			return 1
		else:
			if verbose:
				print('out of range')
				#import pdb; pdb.set_trace()
			return 0
	def get_t(self, zonelist_src, zonelist_dst, ts_src, ts_dst, verbose=False):
		#return   camId_src, camId_dst, zonelists_src, zonelists_dst, t_src, t_dst
		assert zonelist_src in self.zonelists_src and zonelist_dst in self.zonelists_dst
		t_src = ts_src[zonelist_src.index(self.z_src)]
		t_dst = ts_dst[zonelist_dst.index(self.z_dst)]
		return t_src, t_dst

	def remove_outlier(self, method, lower, upper):
		# calculate dt_min and dt_max from distribution
		dts = [x[1] for x in self.distribution] # sort by dt
		dts.sort()
		if method == 0:		
			# remove fix numbers of min/max (possibly outlier)
			# for example, lower, upper = 3, 3
			if len(dts) < lower+upper:
				return
			dts = dts[lower:-upper]
			self.dt_min = dts[0]
			self.dt_max = dts[-1] + 1
		elif method == 1:
			if len(dts) <= lower+upper:
				return
			# adaptively remove at most of min/max (possibly outlier)
			num_remove_lower = 0
			num_remove_upper = 0
			while num_remove_upper < upper or num_remove_lower < lower:
				remove = False
				w = dts[-2] - dts[1]
				if dts[-1] - dts[-2] > w and num_remove_upper < upper:
					dts = dts[:-1]
					num_remove_upper += 1
					remove = True
				if dts[1] - dts[0] > w and num_remove_lower < lower:
					dts = dts[1:]
					num_remove_lower += 1
					remove = True
				if not remove:
					break
			self.dt_min = dts[0]
			self.dt_max = dts[-1] + 1


		else:
			# remove by cumulative percentage
			# for example, lower, upper = 0.05, 0.05
			if len(dts) < 6:
				return
			n_lower = int(len(dts)*lower+0.5)
			n_upper = int(len(dts)*(1-upper)+0.5)
			if n_lower < n_upper:
				dts = dts[n_lower:-n_lower]
				self.dt_min = dts[0]
				self.dt_max = dts[-1] + 1



class CameraTransitionMerge:
	def __init__(self):
		self.cameras = {} # {camId: camera}
		self.transitions = {} # {(camId_src, camId_dst): [transition]}

	def read_camera_zone_xml(self, filename_pattern, camIds):
		assert len(self.cameras) == 0 # read only once !!!!!!
		for camId in camIds:
			camera = Camera(camId)
			camera.read_zone_xml(filename_pattern % camId)
			self.cameras[camId] = camera

	def read_camera_zone(self, filename):
		assert len(self.cameras) == 0 # read only once !!!!!!
		with open(filename, 'r') as f:
			for line in f:
				row = [int(s) for s in line.split(' ')]
				camId = row[0]
				name = row[1]
				x = row[2]
				y = row[3]
				w = row[4]
				h = row[5]
				if camId not in self.cameras:
					self.cameras[camId] = Camera(camId)
				self.cameras[camId].add_zone(name, x, y, w, h)

	def read_camera_zonelist(self, filename):
		assert len(self.cameras) > 0 # read only once !!!!!!
		with open(filename, 'r') as f:
			for line in f:
				row = [s for s in line.split(' ')]
				camId = int(row[0])
				zonelists = parse_zonelists(row[1])
				assert camId in self.cameras
				for zonelist in zonelists:
					self.cameras[camId].add_zonelist(zonelist)

	def read_camera_timestamp(self, filename):
		with open(filename, 'r') as f:
			for line in f:
				row = [int(s) for s in line.split(' ')]
				camId = row[0]
				t = row[1]
				dt = row[2]
				self.cameras[camId].add_timestamp(t, dt)

	def write_camera_timestamp(self, filename):
		with open(filename, 'w') as f:
			for camId, camera in sorted(self.cameras.items()):
				for (t, dt) in camera.timestamps:
					line = '%d %d %d\n' % (camId, t, dt)
					f.write(line)

	def write_camera_zone(self, filename):
		with open(filename, 'w') as f:
			for camId, camera in sorted(self.cameras.items()):
				for name, (x, y, w, h) in sorted(camera.zones.items()):
					line = '%d %d %d %d %d %d\n' % (camId, name, x, y, w, h)
					f.write(line)

	def write_camera_zonelist(self, filename):
		with open(filename, 'w') as f:
			for camId, camera in sorted(self.cameras.items()):
				# put in the same line
				#line = '%d %s\n' % (camId, zonelist_to_str(camera.zonelists))
				#f.write(line)
				# put in each line
				for zonelist in camera.zonelists:
					line = '%d %s\n' % (camId, zonelist_to_str(zonelist))
					f.write(line)

	def add_camera_zonelist(self, camId, zonelist):
		assert camId in self.cameras
		self.cameras[camId].add_zonelist(zonelist)

	def add_transition(self, camId_src, zonelists_src, camId_dst, zonelists_dst, z_src, z_dst, dt_min, dt_max, merge=True):
		# add new transition
		assert camId_src in self.cameras and camId_dst in self.cameras
		for zonelist in zonelists_src:
			assert z_src in zonelist
			assert zonelist in self.cameras[camId_src].zonelists
		for zonelist in zonelists_dst:
			assert z_dst in zonelist
			assert zonelist in self.cameras[camId_dst].zonelists
		if (camId_src, camId_dst) not in self.transitions:
			self.transitions[(camId_src, camId_dst)] = []
		# TODO: check conflicts with existing transition
		if merge:
			transition = Transition(camId_src, camId_dst)
			transition.set_zonelists(zonelists_src, zonelists_dst, z_src, z_dst, dt_min, dt_max)
			self.transitions[(camId_src, camId_dst)].append(transition)
		else:
			for zonelist_src in zonelists_src:
				for zonelist_dst in zonelists_dst:
					transition = Transition(camId_src, camId_dst)
					transition.set_zonelists([zonelist_src], [zonelist_dst], z_src, z_dst, dt_min, dt_max)
					self.transitions[(camId_src, camId_dst)].append(transition)

	def update_transition(self, camId_src, camId_dst, track_src, track_dst, verbose=False):
		# update transition's dt_min and dt_max
		assert camId_src in self.cameras and camId_dst in self.cameras
		assert camId_src < camId_dst # transition time is uniquely defined between camera pair
		carId_src = -1
		t_src = -math.inf
		for tr in track_src:
			assert tr[0] == camId_src
			if carId_src == -1:
				carId_src = tr[1]
			else:
				assert tr[1] == carId_src
			if CHECK_TRACK_ORDER:
				assert tr[2] > t_src
				t_src = tr[2]
		carId_dst = -1
		t_dst = -math.inf
		for tr in track_dst:
			assert tr[0] == camId_dst
			if carId_dst == -1:
				carId_dst = tr[1]
			else:
				assert tr[1] == carId_dst
			if CHECK_TRACK_ORDER:
				assert tr[2] > t_dst
				t_dst = tr[2]

		if (camId_src, camId_dst) not in self.transitions:
			return -1
		zonelist_src, ts_src = self.cameras[camId_src].match_zonelist_track(track_src, verbose)
		zonelist_dst, ts_dst = self.cameras[camId_dst].match_zonelist_track(track_dst, verbose)

		for transition in self.transitions[(camId_src, camId_dst)]:
			flag = transition.update_transition(zonelist_src, zonelist_dst, ts_src, ts_dst, verbose)
			if flag == 1:
				return 1
			elif flag == -2:
				continue
			elif flag == -3:
				return -3
		return -2

	def remove_outlier(self, method, lower, upper):
		for key, transitions in self.transitions.items():
			for transition in transitions:
				transition.remove_outlier(method, lower, upper)

	def in_transition(self, camId_src, camId_dst, track_src, track_dst, w_min=0, w_max=0, verbose=False, returninfo=False):
		# camId: should be int
		# track: [(camId, carId, t, x, y, w, h)], all are int
		# w_min, w_max: additional tolerance margins, usually non-negative
		assert camId_src in self.cameras and camId_dst in self.cameras
		assert camId_src < camId_dst # transition time is uniquely defined between camera pair
		if verbose:
			print('(camId_src, camId_dst): (%d, %d)'%(camId_src, camId_dst))
		if (camId_src, camId_dst) not in self.transitions:
			# there is no transition time for the camera pair
			return -1

		carId_src = -1
		t_src = -math.inf
		for tr in track_src:
			assert tr[0] == camId_src
			if carId_src == -1:
				carId_src = tr[1]
			else:
				assert tr[1] == carId_src
			if CHECK_TRACK_ORDER:
				assert tr[2] > t_src
				t_src = tr[2]
		carId_dst = -1
		t_dst = -math.inf
		for tr in track_dst:
			assert tr[0] == camId_dst
			if carId_dst == -1:
				carId_dst = tr[1]
			else:
				assert tr[1] == carId_dst
			if CHECK_TRACK_ORDER:
				assert tr[2] > t_dst
				t_dst = tr[2]
		
		zonelist_src, ts_src = self.cameras[camId_src].match_zonelist_track(track_src, verbose)
		zonelist_dst, ts_dst = self.cameras[camId_dst].match_zonelist_track(track_dst, verbose)

		for k, transition in enumerate(self.transitions[(camId_src, camId_dst)]):
			flag = transition.in_transition(zonelist_src, zonelist_dst, ts_src, ts_dst, w_min, w_max, verbose)
			if flag == 1:
				if returninfo:
					t_src, t_dst = transition.get_t(zonelist_src, zonelist_dst, ts_src, ts_dst)
					return k, t_src, t_dst # the k'th transition of (camId_src, camId_dst)
				else:
					return 1
			elif flag == 0:
				return 0
			elif flag == -2:
				continue
			elif flag == -3:
				return -3
		return -2

	def read_transition(self, filename):
		# must read camera zone first !!!!!!
		assert len(self.cameras) > 0
		self.transitions = {}
		with open(filename, 'r') as f:
			for line in f:
				row = line.split(' ')
				camId_src = int(row[0])
				zonelists_src = parse_zonelists(row[1])
				camId_dst = int(row[2])
				zonelists_dst = parse_zonelists(row[3])
				#for zonelist_src in zonelists_src:
				#	self.add_camera_zonelist(camId_src, zonelist_src)
				#for zonelist_dst in zonelists_dst:
				#	self.add_camera_zonelist(camId_dst, zonelist_dst)
				iz_src = int(row[4])
				iz_dst = int(row[5])
				dt_min = int(row[6])
				dt_max = int(row[7])
				self.add_transition(camId_src, zonelists_src, camId_dst, zonelists_dst, iz_src, iz_dst, dt_min, dt_max)

	def write_transition(self, filename):
		with open(filename, 'w') as f:
			for (camId_src, camId_dst) in sorted(self.transitions):
				for transition in self.transitions[(camId_src, camId_dst)]:#sorted(self.transitions[(camId_src, camId_dst)]):
					line = '%d %s %d %s %d %d %d %d\n' % (camId_src, zonelists_to_str(transition.zonelists_src), camId_dst, zonelists_to_str(transition.zonelists_dst),
						transition.z_src, transition.z_dst, transition.dt_min, transition.dt_max)
					f.write(line)

	def write_transition_distribution(self, filename):
		with open(filename, 'w') as f:
			for (camId_src, camId_dst) in sorted(self.transitions):
				for transition in self.transitions[(camId_src, camId_dst)]:#sorted(self.transitions[(camId_src, camId_dst)]):
					for (t_src, dt) in transition.distribution:
						line = '%d %s %d %s %d %d %d %d\n' % (camId_src, zonelists_to_str(transition.zonelists_src), camId_dst, zonelists_to_str(transition.zonelists_dst),
							transition.z_src, transition.z_dst, t_src, dt)
						f.write(line)

	def print_cameras(self):
		print('cameras in CameraTransition:')
		for camId in sorted(self.cameras):
			camera = self.cameras[camId]
			print(camera.id)
			print(camera.zones)
			print(camera.zonelists)
			print(camera.timestamps)

	def print_transitions(self):
		print('transitions in CameraTransition:')
		for (camId_src, camId_dst) in sorted(self.transitions):
			for transition in self.transitions[(camId_src, camId_dst)]:#sorted(self.transitions[(camId_src, camId_dst)]):
				line = '%d %s %d %s %d %d %d %d\n' % (camId_src, zonelists_to_str(transition.zonelists_src), camId_dst, zonelists_to_str(transition.zonelists_dst),
					transition.z_src, transition.z_dst, transition.dt_min, transition.dt_max)
				print(line)

	def print_transitions_distribution(self):
		print('transitions distribution in CameraTransition:')
		for (camId_src, camId_dst) in sorted(self.transitions):
			for transition in self.transitions[(camId_src, camId_dst)]:#sorted(self.transitions[(camId_src, camId_dst)]):
				for (t_src, dt) in transition.distribution:
					line = '%d %s %d %s %d %d %d %d\n' % (camId_src, zonelists_to_str(transition.zonelists_src), camId_dst, zonelists_to_str(transition.zonelists_dst),
						transition.z_src, transition.z_dst, t_src, dt)
					print(line)
	