# 2019-CVPR-AIC-Track-1-UWIPL
## 1. Single Camera Tracking
Please download the source code and follow the instruction from https://github.com/GaoangW/TNT/tree/master/AIC19. <br />

## 2. Deep Feature Re_identification
Please download the source code and follow the instruction from https://github.com/ipl-uw/2019-CVPR-AIC-Track-2-UWIPL. 
Create video2img folder in the downloaded project (i.e., Video-Person-ReID/video2img/).
Put and run `python crop_img.py` in the same folder in the downloaded dataset (i.e., aic19-track1-mtmc/test). You need to creat a folder track1_test_img in the same path (i.e., aic19-track1-mtmc/test/track1_test_img). After that, create a folder track1_sct_img_test_big and run `python crop_img_big.py`. Then, create a folder log in the dowanloaded project (i.e., Video-Person-ReID/log) and put the downloaded model file of track1 ReID in this folder. Finally, run `python Graph_ModelDataGen.py` to obtain the feature files (q_camids3_no_nms_big0510.npy, qf3_no_nms_big0510.npy and q_pids3_no_nms_big0510.npy).<br />
The code is based on Jiyang Gao's Video-Person-ReID \[[code](https://github.com/jiyanggao/Video-Person-ReID)\].<br/>

## 3. Trajectory-Based Camera Link Models
Put the feature files (q_camids3_no_nms_big0510.npy, qf3_no_nms_big0510.npy and q_pids3_no_nms_big0510.npy) in the Transition-Model folder of this project. Then, run `python main_in_transition_matrix.py`. Then, find the results in Transition-Model/transition_data /ICT-no_nms_big510/ and select ict_greedy_iter1550_10.079_643_494.txt (1550 means the index of iteration, 10.079 represents the distance of embeddings, 643 means the number of globelID, 494 is the non-redundant number of globelID). <br />

## 4. NMS
ict_greedy_iter1600_10.559_654_498.txt is the input of NMS so that ict_greedy_iter1600_10.559_654_498.txt needs to be placed in the NMS folder. Run `python NMS_filter.py` to get ict_greedy_iter1600_10.559_654_498_big.txt which is the final result of Track1. <br />
