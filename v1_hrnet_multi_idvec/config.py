import numpy as np 

# headnet 
head_layernum = 1
head_chn = 32

# upsmaple 
upsample_layers = 1
upsample_chn = 32

# size
inp_size = 512 
out_size = 256
base_sigma = 2.0
num_pts = 17
feat_dim = 32
inp_scales = [512, 256, 128]

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

COCO_index = np.int64([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
COCO_reorder = np.int64([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

MPII_index = np.int64([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
MPII_reorder = np.int64([16,14,12,11,13,15,17,18,19,20,10,8,6,5,7,9])

PT_index = np.int64([0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,21])
PT_reorder = np.int64([0,21,20,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

# extra 
max_inst = 30
distributed = True
use_subset = True
