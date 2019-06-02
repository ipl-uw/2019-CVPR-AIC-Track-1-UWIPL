
# coding: utf-8

# In[1]:


import numpy as np
import track_lib

# remove tracks with time less than 3 frames
len_lb = 3

# enlarge box by 40 pixels
box_extend = 40

# nms threshold
filter_thresh = 0.5

# touch boundary margin
boundary_margin = 20

# remove box smaller than 1000
small_area = 1000

# number of frames used in the regression 
fit_len = 10

# camera list for no nms, for example
non_nms_cam_list = []

img_size = {"6":[1920,1080],
           "7":[1920,1080],
           "8":[1920,1080],
           "9":[1920,1080],
           "10":[1920,1080],
           "16":[1920,1080],
           "17":[1920,1080],
           "18":[2560,1920],
           "19":[2560,1920],
           "20":[2560,1920],
           "21":[1920,1080],
           "22":[1920,1080],
           "23":[2560,1920],
           "24":[2560,1920],
           "25":[2560,1920],
           "26":[1920,1080],
           "27":[1920,1080],
           "28":[1920,1080],
           "29":[1920,1080],
           "33":[1920,1080],
           "34":[1920,1080],
           "35":[1920,1080],
           "36":[1920,1080]}
txt_file = np.loadtxt("ict_greedy_iter1600_10.559_654_498.txt",dtype='float',delimiter=' ')


# In[2]:


def enlarge_box(orig_M, img_size, box_extend):
    new_M = orig_M.copy()
    for key in img_size.keys():
        #print(key)
        img_shape = img_size[key]
        M = orig_M[orig_M[:,0]==float(key),:].copy()
    
        xmin = M[:,3].copy()
        ymin = M[:,4].copy()
        w = M[:,5].copy()
        h = M[:,6].copy()
        xmax = xmin+w
        ymax = ymin+h
        xmin = np.maximum(xmin-box_extend/2,1)
        ymin = np.maximum(ymin-box_extend/2,1)
        xmax = np.minimum(xmax+box_extend/2,img_shape[0])
        ymax = np.minimum(ymax+box_extend/2,img_shape[1])
        w = xmax-xmin
        h = ymax-ymin
        #import pdb; pdb.set_trace()
    
        new_M[new_M[:,0]==float(key),3] = xmin
        new_M[new_M[:,0]==float(key),4] = ymin
        new_M[new_M[:,0]==float(key),5] = w
        new_M[new_M[:,0]==float(key),6] = h
    return new_M


# In[3]:


# remove small
area_vec = txt_file[:,5]*txt_file[:,6]
small_idx = np.where(area_vec<small_area)[0]
#import pdb; pdb.set_trace()
txt_file = np.delete(txt_file, small_idx, axis=0)

large_M = enlarge_box(txt_file, img_size, box_extend)
#import pdb; pdb.set_trace()

refine_M = []
# nms
for key in img_size.keys():
    print(key)
    
    img_shape = img_size[key]
    small_M = txt_file[txt_file[:,0]==float(key),:].copy()
    M = large_M[large_M[:,0]==float(key),:].copy()
    
    if key in non_nms_cam_list:
        #import pdb; pdb.set_trace()
        if len(refine_M)==0:
            refine_M = M
        else:
            refine_M = np.concatenate((refine_M,M),axis=0)
        continue
        
    frs = np.unique(M[:,2])
    remove_rows = []
    for n in range(len(frs)):
        fr_id = frs[n]
        temp_row_idx = np.where(M[:,2]==fr_id)[0]
        temp_M = M[temp_row_idx,:]
        #if len(temp_row_idx)<=1:
        #import pdb; pdb.set_trace()
        overlap_mat,_,_,_ = track_lib.get_overlap(temp_M[:,3:7], temp_M[:,3:7])
        for k in range(len(overlap_mat)):
            overlap_mat[k,k] = 0.0
        idx = np.where(overlap_mat>filter_thresh)
        if len(idx[0])==0:
            continue
        
        #import pdb; pdb.set_trace()
        for k in range(len(idx[0])):
            box1 = temp_M[idx[0][k],3:7]
            box2 = temp_M[idx[1][k],3:7]
            ymax1 = box1[1]+box1[3]
            ymax2 = box2[1]+box2[3]
            if ymax1>ymax2:
                remove_rows.extend([temp_row_idx[idx[1][k]]])
            else:
                remove_rows.extend([temp_row_idx[idx[0][k]]])
        #import pdb; pdb.set_trace()
    remove_rows = list(set(remove_rows))    
    delete_M = np.delete(small_M,remove_rows,axis=0)
    #import pdb; pdb.set_trace()
    if len(refine_M)==0:
        refine_M = delete_M
    else:
        refine_M = np.concatenate((refine_M,delete_M),axis=0)

refine_M1 = refine_M.copy()
refine_M = []
for key in img_size.keys():
    print(key)
    img_shape = img_size[key]
    M = refine_M1[refine_M1[:,0]==float(key),:].copy()
    ids = np.unique(M[:,1])
    remove_rows = []
    for n in range(len(ids)):
        temp_id = ids[n]
        temp_row_idx = np.where(M[:,1]==temp_id)[0]
        track_len = len(temp_row_idx)
        
        # remove small tracks
        if track_len<=len_lb:
            remove_rows.extend(temp_row_idx)
            continue
        
        # remove boundary
        temp_M = M[temp_row_idx,:]
        areas = temp_M[:,5]*temp_M[:,6]
        boundary_check = (temp_M[:,3]<boundary_margin)+(temp_M[:,4]<boundary_margin)            +(np.absolute(temp_M[:,3]+temp_M[:,5]-img_shape[0])<boundary_margin)            +(np.absolute(temp_M[:,4]+temp_M[:,6]-img_shape[1])<boundary_margin)
        boundary_check = boundary_check.astype(int)
        t_vec = temp_M[:,2]
        cand_t_vec = t_vec[boundary_check<0.5]
        if len(cand_t_vec)<2:
            remove_rows.extend(temp_row_idx[boundary_check>0.5])
            continue

        
        t_len = min(len(cand_t_vec), fit_len)
        temp_area = areas[boundary_check<0.5]
        t_vec1 = cand_t_vec[0:t_len]
        area1 = temp_area[0:t_len]
        t_vec2 = cand_t_vec[-t_len:]
        area2 = temp_area[-t_len:]
        
        A1 = np.vstack([t_vec1, np.ones(len(t_vec1))]).T
        k1,b1 = np.linalg.lstsq(A1, area1)[0]
        
        A2 = np.vstack([t_vec2, np.ones(len(t_vec2))]).T
        k2,b2 = np.linalg.lstsq(A2, area2)[0]
        
        bound_t = t_vec[boundary_check>0.5]
        '''
        if len(bound_t)==0:
            continue
        if len(t_vec1)==0 or len(t_vec2)==0:
            import pdb; pdb.set_trace()
        '''
        bound_area = areas[boundary_check>0.5]
        t_dist1 = np.absolute(bound_t-t_vec1[0])
        t_dist2 = np.absolute(bound_t-t_vec2[-1])
        bound_t1 = bound_t[t_dist1<=t_dist2]
        bound_t2 = bound_t[t_dist1>t_dist2]
        bound_area1 = bound_area[t_dist1<=t_dist2]
        bound_area2 = bound_area[t_dist1>t_dist2]
        
        if len(bound_t1)>0:
            pred_area1 = k1*bound_t1+b1
            temp_remove_idx = np.where(bound_area1<pred_area1*2/3)[0]
            
            if len(temp_remove_idx)>0:
                rmv_idx = temp_row_idx[boundary_check>0.5]
                rmv_idx = rmv_idx[t_dist1<=t_dist2]
                rmv_idx = rmv_idx[temp_remove_idx]
                remove_rows.extend(rmv_idx)
                
        if len(bound_t2)>0:
            pred_area2 = k2*bound_t2+b2
            temp_remove_idx = np.where(bound_area2<pred_area2*2/3)[0]
            
            if len(temp_remove_idx)>0:
                rmv_idx = temp_row_idx[boundary_check>0.5]
                rmv_idx = rmv_idx[t_dist1>t_dist2]
                rmv_idx = rmv_idx[temp_remove_idx]
                remove_rows.extend(rmv_idx)
        
    
    delete_M = np.delete(M,remove_rows,axis=0)
    #import pdb; pdb.set_trace()
    if len(refine_M)==0:
        refine_M = delete_M
    else:
        refine_M = np.concatenate((refine_M,delete_M),axis=0)
    #print(int(key))

refine_M = refine_M.astype(int)
np.savetxt('refine.txt',refine_M, fmt='%d', delimiter=' ')

# enlarge box
refine_M = enlarge_box(refine_M, img_size, box_extend)
'''
for key in img_size.keys():
    print(key)
    img_shape = img_size[key]
    M = refine_M[refine_M[:,0]==float(key),:].copy()
    
    xmin = M[:,3].copy()
    ymin = M[:,4].copy()
    w = M[:,5].copy()
    h = M[:,6].copy()
    xmax = xmin+w
    ymax = ymin+h
    xmin = np.maximum(xmin-box_extend/2,1)
    ymin = np.maximum(ymin-box_extend/2,1)
    xmax = np.minimum(xmax+box_extend/2,img_shape[0])
    ymax = np.minimum(ymax+box_extend/2,img_shape[1])
    w = xmax-xmin
    h = ymax-ymin
    #import pdb; pdb.set_trace()
    
    refine_M[refine_M[:,0]==float(key),3] = xmin
    refine_M[refine_M[:,0]==float(key),4] = ymin
    refine_M[refine_M[:,0]==float(key),5] = w
    refine_M[refine_M[:,0]==float(key),6] = h
'''    
np.savetxt('ict_greedy_iter1600_10.559_654_498_big.txt',refine_M, fmt='%d', delimiter=' ')
    

        


# In[9]:


img_size.keys()

