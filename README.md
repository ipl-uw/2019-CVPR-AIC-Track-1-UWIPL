# 2019-CVPR-AIC-Track-1-UWIPL
# 1. Single Camera Tracking
Please download the source code and follow the instruction from https://github.com/GaoangW/TNT/tree/master/AIC19. <br />

# 2. Trajectory-Based Camera Link Models 

# 3. Deep Feature Re_identification
Please download the source code and follow the instruction from https://github.com/ipl-uw/2019-CVPR-AIC-Track-2-UWIPL. <br />
Create video2img folder in the downloaded project (i.e., Video-Person-ReID/video2img/)
Put and run python crop_img.py in the same folder in the downloaded dataset (i.e., aic19-track1-mtmc/test). You need to creat a folder track1_test_img in the same path (i.e., aic19-track1-mtmc/test/track1_test_img). After that, create a folder track1_sct_img_test_big and run python crop_img_big.py. Then, create a folder log in the dowanloaded project and put the downloaded model file of track1 ReID. Finally, run python Graph_ModelDataGen.py to obtain the feature files.

