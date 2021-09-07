import network 
import config 
import torch 

refiner = network.RefineNet(config.refine_dim, config.num_heads, config.pos_embed, config.depth)
x = torch.zeros(1, len(config.inp_scales), config.num_pts, config.top_k_candidates, 132030)
refiner(x)
