import numpy as np 

# size
inp_size = 512 
inp_scales = [512, 256, 128]
out_size = 64
base_sigma = 2.5
num_pts = 17

# point index 
COCO_index = np.int64([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
COCO_reorder = np.int64([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

# augmentation 
rotation = 30
min_scale = 0.5
max_scale = 1.25
max_translate = 50

blur_prob = 0.0
blur_size = [7, 11, 15, 21]
blur_type = ['vertical','horizontal','mean']

# training 
max_iter = 50000
max_epoch = 300
init_lr = 0.001
iter_density_pretrain = 50
decay = 0.0001
momentum = 0.9
lr_step = 35000
lr_epoch = [200,250]
save_interval = 1

# extra 
max_inst = 30
distributed = True
