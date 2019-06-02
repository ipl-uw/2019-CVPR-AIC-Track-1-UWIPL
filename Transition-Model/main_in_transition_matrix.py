'''
Tsung-Wei Huang
AI City 2019
'''

import os
from os import mkdir
from os.path import join, isdir

import numpy as np

from scipy.spatial import distance
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster.hierarchy as sch
from scipy.cluster.vq import vq,kmeans,whiten
import matplotlib.pylab as plt


from itertools import combinations

from camera_transition import CameraTransitionMerge



use_timestamp = True

camIds = [
	6, 7, 8, 9,
	10, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36,
]

'''camSets = [
	[6, 7, 8, 9],
	[10, 16, 17],
	[17, 18, 19, 20],
	[20, 21, 22, 23],
	[23, 24, 25, 26],
	[26, 27, 28, 29],
	[29, 34],
	[33, 34, 35, 36],
]'''

# camId pair to apply transition order
camIds_transitionOrder = {
	(10, 16), (10, 17), (16, 17),
	(17, 18), (17, 19), (17, 20), (18, 19), (18, 20), (19, 20),
	(20, 21), (20, 22), (20, 23), (21, 22), (21, 23), (22, 23),
	(23, 24), (23, 25), (23, 26), (24, 25), (24, 26), (25, 26),
	(26, 27), (26, 28), (26, 29), (27, 28), (27, 29), (28, 29),
	#(33, 35), (33, 36), (34, 35), (34, 36), (35, 36),
	(18, 23), (20, 25), 

}
camIds_transitionOrder = {}

cameraTransition = CameraTransitionMerge()


# read camera transition from file

data_dir = './transition_data'
print('using data_dir: %s'%data_dir)
ICT_folder = 'ICT-no_nms_big510'
ICT_dir = join(data_dir, ICT_folder)
mkdir(ICT_dir)

camera_zone_filename = 'camera_zones.txt'
camera_zonelist_filename = 'camera_zonelists.txt'
camera_timestamp_filename = 'camera_timestamps.txt'
camera_transition_filename = 'camera_transitions.txt'
camera_transition_distribution_filename = 'camera_transitions_distribution.txt'


in_transition_matrix_filename = 'in_transition_matrix.npy'
transition_k_matrix_filename = 'transition_k_matrix.npy'
transition_t_src_matrix_filename = 'transition_t_src_matrix.npy'
transition_t_dst_matrix_filename = 'transition_t_dst_matrix.npy'


ICT_result_filename = 'ict_greedy.txt'
ICT_result_folder = 'ict_greedy'




cameraTransition.read_camera_zone(join(data_dir, camera_zone_filename))
print('camera zones readed from %s' % camera_zone_filename)

cameraTransition.read_camera_zonelist(join(data_dir, camera_zonelist_filename))
print('camera zonelists readed from %s' % camera_zonelist_filename)

if use_timestamp:
	cameraTransition.read_camera_timestamp(join(data_dir, camera_timestamp_filename))
	print('camera timestamp readed from %s' % camera_timestamp_filename)

cameraTransition.read_transition(join(data_dir, camera_transition_filename))
print('camera transition readed from %s' % camera_transition_filename)

#cameraTransition.print_cameras()
#cameraTransition.print_transitions()




# read order of all tracks
camId_filename = './q_camids3_no_nms_big0510.npy' # track 1 train + track 2 train (517 classes)
carId_filename = './q_pids3_no_nms_big0510.npy'
feature_filename = './qf3_no_nms_big0510.npy'
sct_filepattern = './txt_GPS_new/c%03d.txt'

camIds_array = np.load(camId_filename)
carIds_array = np.load(carId_filename)
features = np.load(feature_filename)

assert camIds_array.shape[0] == carIds_array.shape[0]

num_cars = camIds_array.shape[0]

print(camIds_array.shape)
print(carIds_array.shape)
print(features.shape)

### sort by camId

index_array = np.argsort(camIds_array)
camIds_array = camIds_array[index_array]
carIds_array = carIds_array[index_array]
features = features[index_array]


camIdcarId_i = {}
for i in range(num_cars):
	camId = camIds_array[i]
	carId = carIds_array[i]
	camIdcarId_i[(camId, carId)] = i
#import pdb; pdb.set_trace()
#for i in range(num_cars):
#	print((camIds[i], carIds[i]))

# read single camera tracking from file
print('reading single camera tracking from file')
tracks = {}
for camId in camIds:
	tracks[camId] = {}
	sct_filename = sct_filepattern%camId
	print(sct_filename)
	with open(sct_filename, 'r') as f:
		for line in f:
			row = [int(s) for s in line.split(',')[:6]]
			t = row[0]
			carId = row[1]
			x = row[2]
			y = row[3]
			w = row[4]
			h = row[5]
			if carId not in tracks[camId]:
				tracks[camId][carId] = []
			tracks[camId][carId].append((camId, carId, t, x, y, w, h))

# calculate all pairwise in_transition value
read_from_file = False#True
w_min = 0
w_max = 0
if read_from_file:
	in_transition_matrix = np.load(join(data_dir, in_transition_matrix_filename[:-4]+'_%d_%d.npy'%(w_min, w_max)))
	transition_k_matrix = np.load(join(data_dir, transition_k_matrix_filename[:-4]+'_%d_%d.npy'%(w_min, w_max)))
	transition_t_src_matrix = np.load(join(data_dir, transition_t_src_matrix_filename[:-4]+'_%d_%d.npy'%(w_min, w_max)))
	transition_t_dst_matrix = np.load(join(data_dir, transition_t_dst_matrix_filename[:-4]+'_%d_%d.npy'%(w_min, w_max)))
else:
	# for the meaning of the value, see camera_transition.py
	in_transition_matrix = np.zeros((num_cars, num_cars), dtype=np.int32)
	transition_k_matrix = -np.ones((num_cars, num_cars), dtype=np.int32)
	transition_t_src_matrix = -np.ones((num_cars, num_cars), dtype=np.int32)
	transition_t_dst_matrix = -np.ones((num_cars, num_cars), dtype=np.int32)
	for i in range(num_cars):
		if i%100 == 0:
			print('%d/%d'%(i, num_cars))
		camId_src = camIds_array[i]
		carId_src = carIds_array[i]
		track_src = tracks[camId_src][carId_src]
		for j in range(i+1, num_cars):
			#print('(%d, %d)'%(i, j))
			camId_dst = camIds_array[j]
			carId_dst = carIds_array[j]
			track_dst = tracks[camId_dst][carId_dst]
			#import pdb; pdb.set_trace()
			if camId_src < camId_dst:
				flag = cameraTransition.in_transition(camId_src, camId_dst, track_src, track_dst, w_min, w_max, verbose=False)
				#if flag == -2: ######## more strict constraint ######## consider not defined zonelist pair as invalid
				#	flag = 0
				in_transition_matrix[i, j] = flag
				if flag == 1:
					k, t_src, t_dst = cameraTransition.in_transition(camId_src, camId_dst, track_src, track_dst, w_min, w_max, verbose=False, returninfo=True)
					transition_k_matrix[i, j] = k
					transition_t_src_matrix[i, j] = t_src
					transition_t_dst_matrix[i, j] = t_dst
	in_transition_matrix = in_transition_matrix + np.transpose(in_transition_matrix)
	transition_k_matrix = transition_k_matrix + np.transpose(transition_k_matrix)
	transition_t_src_matrix = transition_t_src_matrix + np.transpose(transition_t_src_matrix)
	transition_t_dst_matrix = transition_t_dst_matrix + np.transpose(transition_t_dst_matrix)
	# save in_transition_matrix to file
	np.save(join(data_dir, in_transition_matrix_filename[:-4]+'_%d_%d.npy'%(w_min, w_max)), in_transition_matrix)
	np.save(join(data_dir, transition_k_matrix_filename[:-4]+'_%d_%d.npy'%(w_min, w_max)), transition_k_matrix)
	np.save(join(data_dir, transition_t_src_matrix_filename[:-4]+'_%d_%d.npy'%(w_min, w_max)), transition_t_src_matrix)
	np.save(join(data_dir, transition_t_dst_matrix_filename[:-4]+'_%d_%d.npy'%(w_min, w_max)), transition_t_dst_matrix)

# calculate zonelist_src, ts_src = self.cameras[camId_src].match_zonelist_track(track_src, verbose)
print('start calculating (zonelist, ts) for each track')
zonelist_list = []
ts_list = []
for i in range(num_cars):
	camId_src = camIds_array[i]
	carId_src = carIds_array[i]
	track_src = tracks[camId_src][carId_src]
	zonelist_src, ts_src = cameraTransition.cameras[camId_src].match_zonelist_track(track_src)
	zonelist_list.append(zonelist_src)
	ts_list.append(ts_src)
print('finish calculating (zonelist, ts) for each track')

# create transition_map

class TransitionOrder:
	def __init__(self, camId_src, camId_dst, k):
		assert camId_src < camId_dst
		self.camId_src = camId_src
		self.camId_dst = camId_dst
		self.k = k
		self.it_src = set()
		self.it_dst = set()
		self.ij_candidates = set() # keep updated possible ij candidate
		self.ij_pairs = set() # keep matched ij pair
		self.i_src = [] # sorted list
		self.i_dst = [] # sorted list

	def add_candidate(self, i, j, t_src, t_dst):
		self.it_src.add((i, t_src))
		self.it_dst.add((j, t_dst))
		self.ij_candidates.add((i, j))

	def sort_by_t(self):
		self.i_src = [x[0] for x in sorted(self.it_src, key=lambda x: x[1])]
		self.i_dst = [x[0] for x in sorted(self.it_dst, key=lambda x: x[1])]

	def add_pair(self, i, j):
		assert (i, j) in self.ij_candidates
		self.ij_pairs.add((i, j))
		self.ij_candidates.remove((i, j))

		cross_pairs = set()
		for (ki, kj) in self.ij_candidates:
			if ki == i or kj == j:
				cross_pairs.add((ki, kj))
			elif self.i_src.index(ki) < self.i_src.index(i) and self.i_dst.index(kj) > self.i_dst.index(j):
				cross_pairs.add((ki, kj))
			elif self.i_src.index(ki) > self.i_src.index(i) and self.i_dst.index(kj) < self.i_dst.index(j):
				cross_pairs.add((ki, kj))
		for (ki, kj) in cross_pairs:
			self.ij_candidates.remove((ki, kj))
		return cross_pairs

transitionOrders = {}
for i in range(num_cars):
	for j in range(num_cars):
		camId_src = camIds_array[i]
		camId_dst = camIds_array[j]
		if camId_src < camId_dst and in_transition_matrix[i, j] == 1:
			k = transition_k_matrix[i, j]
			t_src = transition_t_src_matrix[i, j]
			t_dst = transition_t_dst_matrix[i, j]
			if (camId_src, camId_dst, k) not in transitionOrders:
				transitionOrders[(camId_src, camId_dst, k)] = TransitionOrder(camId_src, camId_dst, k)
			transitionOrder = transitionOrders[(camId_src, camId_dst, k)]
			transitionOrder.add_candidate(i, j, t_src, t_dst)

for (camId_src, camId_dst, k), transitionOrder in transitionOrders.items():
	transitionOrder.sort_by_t()

#########################	clustering


###### greedy clustering

# assert the camIds_array is sorted
camId = 0
for i in range(num_cars):
	assert camIds_array[i] >= camId
	camId = camIds_array[i]


print('start calculating distance matrix')
distmat = distance.cdist(features, features, 'euclidean')
print('finish calculating distance matrix')


## greedy
gId_i = {} # {gId: [i]}
gId_in_transition = {} # {gId: np.array()}
gIds_array = -np.ones(num_cars, dtype=np.int32)
num_gId = 0
dist_thresh = 30

# sort distmat
print('start sorting distance matrix')
index_sort = np.argsort(distmat, axis=None)
print('finish sorting distance matrix')
i_sort, j_sort = np.unravel_index(index_sort, (num_cars, num_cars))
idd = 0

niter = 0
niter_max = 2500
while niter < niter_max:
	print('niter: %d'%niter)

	while in_transition_matrix[i_sort[idd], j_sort[idd]] != 1:# or camIds_array[i_sort[idd]] >= camIds_array[j_sort[idd]]:
		idd += 1
		if not idd < num_cars*num_cars:
			print('no more possible connection')
			break
	if not idd < num_cars*num_cars:
		break
	i_min = i_sort[idd]
	j_min = j_sort[idd]
	idd += 1
	dist_min = distmat[i_min, j_min]

	if dist_min > dist_thresh:
		print('no small enough distance')
		break
	print('dist_min: %f'%dist_min)
	
	gId = -1
	if gIds_array[i_min] < 0 and gIds_array[j_min] < 0:
		# non have been assigned gId
		gId = num_gId
		print('new gId %d'%gId)
		gIds_array[i_min] = gId
		gIds_array[j_min] = gId
		gId_i[gId] = [i_min, j_min]
		gId_in_transition[gId] = -np.ones(num_cars, dtype=np.int32)
		for k in range(num_cars):
			if in_transition_matrix[i_min, k] == 0 or in_transition_matrix[j_min, k] == 0:
				gId_in_transition[gId][k] = 0
		num_gId += 1
	elif (gIds_array[i_min] >= 0 and gIds_array[j_min] < 0) or (gIds_array[i_min] < 0 and gIds_array[j_min] >= 0):
		# one has been assigned
		if gIds_array[i_min] < 0 and gIds_array[j_min] >= 0:
			i_min, j_min = j_min, i_min
		gId = gIds_array[i_min]
		print('append to gId %d'%gId)
		gIds_array[j_min] = gId
		gId_i[gId].append(j_min)
		for k in range(num_cars):
			if in_transition_matrix[j_min, k] == 0:
				gId_in_transition[gId][k] = 0
	elif gIds_array[i_min] != gIds_array[j_min]:
		# both have been assigned
		gId = gIds_array[i_min]
		gId_old = gIds_array[j_min]
		print('merge gId %d to gId %d'%(gId_old, gId))
		for j in gId_i[gId_old]:
			gIds_array[j] = gId
		gId_i[gId].extend(gId_i[gId_old])
		gId_i.pop(gId_old)
		for k in range(num_cars):
			if gId_in_transition[gId_old][k] == 0:
				gId_in_transition[gId][k] = 0
		gId_in_transition.pop(gId_old)

	#### dynamicly modify in_transition_matrix

	# update conflicts of gId
	if gId >= 0:
		print('%d: %s'%(gId, gId_i[gId]))
		print('-'.join([str((camIds_array[i], carIds_array[i])) for i in sorted(gId_i[gId])]))
		for i in gId_i[gId]:
			for k in range(num_cars):
				if gId_in_transition[gId][k] == 0:
					in_transition_matrix[i, k] = 0
					in_transition_matrix[k, i] = 0 # i pair
					if gIds_array[k] >= 0:
						gId_in_transition[gIds_array[k]][i] = 0 # gId pair

		for i in gId_i[gId]:
			for j in gId_i[gId]:
				in_transition_matrix[i, j] = -4

	# transition order
	camId_src = camIds_array[i_min]
	camId_dst = camIds_array[j_min]
	k = transition_k_matrix[i_min, j_min]
	#assert camId_src < camId_dst
	if camId_src >= camId_dst:
		camId_src, camId_dst = camId_dst, camId_src
		i_min, j_min = j_min, i_min
	if (camId_src, camId_dst) in camIds_transitionOrder:
		print('apply transition order')
		cross_pairs = transitionOrders[(camId_src, camId_dst, k)].add_pair(i_min, j_min)
		for (i, j) in cross_pairs:
			in_transition_matrix[i, j] = 0
			if gIds_array[i] >= 0:
				gId_in_transition[gIds_array[i]][j] = 0
				for k in gId_i[gIds_array[i]]:
					in_transition_matrix[k, j] = 0
					in_transition_matrix[j, k] = 0
			in_transition_matrix[j, i] = 0
			if gIds_array[j] >= 0:
				gId_in_transition[gIds_array[j]][i] = 0
				for k in gId_i[gIds_array[j]]:
					in_transition_matrix[k, i] = 0
					in_transition_matrix[i, k] = 0

	niter += 1

	if niter > 1200 and niter % 50 == 0:
		labels = np.copy(gIds_array)
		num_gId_all = num_gId
		for i in range(num_cars):
			if labels[i] == -1:
				labels[i] = num_gId_all
				num_gId_all += 1
		np.save(join(ICT_dir, 'labels_iter%d_%.3f_%d_%d.npy'%(niter, dist_min, num_gId, len(gId_i))), labels)

		# calculate some statistics

		carIdglobal_camIdcarId = {}
		globalIds = {}
		for i in range(num_cars):
			carId_global = labels[i]
			camId = camIds_array[i]
			carId = carIds_array[i]
			if carId_global not in carIdglobal_camIdcarId:
				carIdglobal_camIdcarId[carId_global] = []
			carIdglobal_camIdcarId[carId_global].append((camId, carId))
			if camId not in globalIds:
				globalIds[camId] = {}
			globalIds[camId][carId] = carId_global


		# write ICT result

		with open(join(ICT_dir, ICT_result_filename[:-4]+'_iter%d_%.3f_%d_%d.txt'%(niter, dist_min, num_gId, len(gId_i))), 'w') as f:
			for camId in sorted(globalIds):
				for carId in sorted(globalIds[camId]):
					for track in tracks[camId][carId]:
						f.write('%d %d %d %d %d %d %d -1 -1\n' % (camId, globalIds[camId][carId], track[2], track[3], track[4], track[5], track[6]))
		with open(join(ICT_dir, ICT_result_filename[:-4]+'_iter%d_%.3f_%d_%d-sortbycar.txt'%(niter, dist_min, num_gId, len(gId_i))), 'w') as f:
			for carId_global, camIdcarIds in carIdglobal_camIdcarId.items():
				for camId, carId in camIdcarIds:
					for track in tracks[camId][carId]:
						f.write('%d %d %d %d %d %d %d -1 -1\n' % (camId, carId_global, track[2], track[3], track[4], track[5], track[6]))
		ICT_result_dir = join(ICT_dir, ICT_result_folder+'_iter%d_%.3f_%d_%d'%(niter, dist_min, num_gId, len(gId_i)))
		mkdir(ICT_result_dir)
		for camId in sorted(globalIds):
			with open(join(ICT_result_dir, 'c%03d_test.txt'%camId), 'w') as f:
				for carId in sorted(globalIds[camId]):
					for track in tracks[camId][carId]:
						f.write('%d,%d,%d,%d,%d,%d,-1,-1,-1\n' % (track[2], globalIds[camId][carId], track[3], track[4], track[5], track[6]))


		# show result and write result
		f = open(join(ICT_dir, 'carIdglobal_camIdcarId_iter%d_%.3f_%d_%d.txt'%(niter, dist_min, num_gId, len(gId_i))), 'w')
		for carIdglobal, camIdcarIds in carIdglobal_camIdcarId.items():
			dists = []
			comb = combinations(camIdcarIds, 2)
			for (camId_src, carId_src), (camId_dst, carId_dst) in comb:
				if distmat.shape == (num_cars, num_cars):
					dists.append(distmat[camIdcarId_i[(camId_src, carId_src)], camIdcarId_i[(camId_dst, carId_dst)]])
				else: #### for spectral culstering 
					i = camIdcarId_i[(camId_src, carId_src)]
					j = camIdcarId_i[(camId_dst, carId_dst)]
					d = int(i*num_cars + j - (i+1)*(i+2)/2)
					dists.append(distmat[d])
			distavg = sum(dists)/len(dists) if len(dists) > 0 else 0

			zonelist_tss = []
			for (camId_src, carId_src) in camIdcarIds:
				i = camIdcarId_i[(camId_src, carId_src)]
				zonelist_tss.append((camId_src, zonelist_list[i], ts_list[i]))
			zonelist_tss.sort(key=lambda x: x[2])
			#print('%d: %f' % (carIdglobal, distavg))
			#print(camIdcarIds)
			#print(zonelist_tss)
			f.write('%d: %f\n' % (carIdglobal, distavg))
			f.write(str(camIdcarIds)+'\n')
			f.write(str(zonelist_tss)+'\n')
		f.close()



# create label array
gId_i_all = gId_i.copy()
labels = np.copy(gIds_array)
num_gId_all = num_gId
for i in range(num_cars):
	if labels[i] == -1:
		labels[i] = num_gId_all
		gId_i_all[num_gId_all] = [i]
		num_gId_all += 1
np.save(join(ICT_dir, 'labels_iter%d_%.3f_%d_%d.npy'%(niter, dist_min, num_gId, len(gId_i))), labels)


# calculate some statistics

carIdglobal_camIdcarId = {}
globalIds = {}
for i in range(num_cars):
	carId_global = labels[i]
	camId = camIds_array[i]
	carId = carIds_array[i]
	if carId_global not in carIdglobal_camIdcarId:
		carIdglobal_camIdcarId[carId_global] = []
	carIdglobal_camIdcarId[carId_global].append((camId, carId))
	if camId not in globalIds:
		globalIds[camId] = {}
	globalIds[camId][carId] = carId_global


# write ICT result

with open(join(ICT_dir, ICT_result_filename[:-4]+'_iter%d_%.3f_%d_%d.txt'%(niter, dist_min, num_gId, len(gId_i))), 'w') as f:
	for camId in sorted(globalIds):
		for carId in sorted(globalIds[camId]):
			for track in tracks[camId][carId]:
				f.write('%d %d %d %d %d %d %d -1 -1\n' % (camId, globalIds[camId][carId], track[2], track[3], track[4], track[5], track[6]))
with open(join(ICT_dir, ICT_result_filename[:-4]+'_iter%d_%.3f_%d_%d-sortbycar.txt'%(niter, dist_min, num_gId, len(gId_i))), 'w') as f:
	for carId_global, camIdcarIds in carIdglobal_camIdcarId.items():
		for camId, carId in camIdcarIds:
			for track in tracks[camId][carId]:
				f.write('%d %d %d %d %d %d %d -1 -1\n' % (camId, carId_global, track[2], track[3], track[4], track[5], track[6]))
ICT_result_dir = join(ICT_dir, ICT_result_folder+'_iter%d_%.3f_%d_%d'%(niter, dist_min, num_gId, len(gId_i)))
mkdir(ICT_result_dir)
for camId in sorted(globalIds):
	with open(join(ICT_result_dir, 'c%03d_test.txt'%camId), 'w') as f:
		for carId in sorted(globalIds[camId]):
			for track in tracks[camId][carId]:
				f.write('%d,%d,%d,%d,%d,%d,-1,-1,-1\n' % (track[2], globalIds[camId][carId], track[3], track[4], track[5], track[6]))

# show result and write result
f = open(join(ICT_dir, 'carIdglobal_camIdcarId_iter%d_%.3f_%d_%d.txt'%(niter, dist_min, num_gId, len(gId_i))), 'w')
for carIdglobal, camIdcarIds in carIdglobal_camIdcarId.items():
	dists = []
	comb = combinations(camIdcarIds, 2)
	for (camId_src, carId_src), (camId_dst, carId_dst) in comb:
		if distmat.shape == (num_cars, num_cars):
			dists.append(distmat[camIdcarId_i[(camId_src, carId_src)], camIdcarId_i[(camId_dst, carId_dst)]])
		else: #### for spectral culstering 
			i = camIdcarId_i[(camId_src, carId_src)]
			j = camIdcarId_i[(camId_dst, carId_dst)]
			d = int(i*num_cars + j - (i+1)*(i+2)/2)
			dists.append(distmat[d])
	distavg = sum(dists)/len(dists) if len(dists) > 0 else 0

	zonelist_tss = []
	for (camId_src, carId_src) in camIdcarIds:
		i = camIdcarId_i[(camId_src, carId_src)]
		zonelist_tss.append((camId_src, zonelist_list[i], ts_list[i]))
	zonelist_tss.sort(key=lambda x: x[2])
	print('%d: %f' % (carIdglobal, distavg))
	print(camIdcarIds)
	print(zonelist_tss)
	f.write('%d: %f\n' % (carIdglobal, distavg))
	f.write(str(camIdcarIds)+'\n')
	f.write(str(zonelist_tss)+'\n')
f.close()




