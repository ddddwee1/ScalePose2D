import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import torch.distributed as dist
import torch
import torch.nn as nn 
import torch.multiprocessing as mp 
import config 
import network 
import datareader 
import numpy as np 
from TorchSUL import Model as M
import loss 
from time import gmtime, strftime
import random 
import visutil 

def main():
	ngpus_per_node = torch.cuda.device_count()
	worldsize = ngpus_per_node
	mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, ))

def main_worker(gpu, ngpus_per_node):
	print('Use GPU:', gpu)
	dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=ngpus_per_node, rank=gpu)
	print('Group initialized.')

	# initialize the network 
	model_dnet = network.MultiScaleNet()
	x = np.float32(np.random.random(size=[1,3,512,512]))
	x = torch.from_numpy(x)
	with torch.no_grad():
		model_dnet(x)
	# input()
	M.Saver(model_dnet.backbone).restore('./imagenet_pretrained_r50/', strict=False)
	model = loss.ModelWithLoss(model_dnet)
	saver = M.Saver(model)
	saver.restore('./model/')

	torch.cuda.set_device(gpu)
	model.cuda(gpu)
	model.eval()
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
	print('Model initialized.')

	# get loader 
	loader, sampler = datareader.get_train_dataloader(8)
	optim = torch.optim.Adam(model.parameters(), lr=config.init_lr)

	for e in range(config.max_epoch):
		print('Replica:%d Epoch:%d'%(gpu, e))
		sampler.set_epoch(e)
		for i, batch in enumerate(loader):
			img = batch[0]
			hmap = batch[-3]
			offset_map = batch[-5]
			pts = batch[-1]
			# if i==0 and gpu==0:
			# 	torch.save([img, hmap, mask, offset_map, offset_weight, pts], 'sampleinp.pth')
			optim.zero_grad()
			hmap_loss, offs_loss, outs, hmap_loss_fused, offs_loss_fused, outs_fused = model(batch)

			ls_large = hmap_loss[0]
			ls_medium = hmap_loss[1]
			ls_small = hmap_loss[2]
			hmap_loss = ls_large + ls_medium + ls_small

			of_large = offs_loss[0]
			of_medium = offs_loss[1]
			of_small = offs_loss[2]
			offs_loss = of_large + of_medium + of_small

			ls_large_f = hmap_loss_fused[0] * 16
			ls_medium_f = hmap_loss_fused[1] * 4
			ls_small_f = hmap_loss_fused[2]
			hmap_loss_f = ls_large_f + ls_medium_f + ls_small_f

			of_large_f = offs_loss_fused[0] 
			of_medium_f = offs_loss_fused[1] 
			of_small_f = offs_loss_fused[2]
			offs_loss_f = of_large_f + of_medium_f + of_small_f

			loss_total = hmap_loss + hmap_loss_f + (offs_loss_f + offs_loss) * 0.01
			loss_total.backward()
			optim.step()
			lr = optim.param_groups[0]['lr']

			if i%100==0 and gpu==0:
				visutil.vis_batch(img, outs[0][:,:1], './outputs/%d_out0.jpg'%i)
				visutil.vis_batch(img, outs[1][:,:1], './outputs/%d_out1.jpg'%i)
				visutil.vis_batch(img, outs[2][:,:1], './outputs/%d_out2.jpg'%i)
				visutil.vis_batch(img, outs_fused[0][:,:1], './outputs/%d_out_f0.jpg'%i)
				visutil.vis_batch(img, outs_fused[1][:,:1], './outputs/%d_out_f1.jpg'%i)
				visutil.vis_batch(img, outs_fused[2][:,:1], './outputs/%d_out_f2.jpg'%i)
				visutil.vis_batch(img, hmap, './outputs/%d_outgt.jpg'%i)
				visutil.vis_offset(img, outs[0][:,1:], pts, './outputs/%d_off0.jpg'%i)
				visutil.vis_offset(img, outs[1][:,1:], pts, './outputs/%d_off1.jpg'%i)
				visutil.vis_offset(img, outs[2][:,1:], pts, './outputs/%d_off2.jpg'%i)
				visutil.vis_offset(img, outs_fused[0][:,1:], pts * config.scales[0], './outputs/%d_off_f0.jpg'%i)
				visutil.vis_offset(img, outs_fused[1][:,1:], pts * config.scales[1], './outputs/%d_off_f1.jpg'%i)
				visutil.vis_offset(img, outs_fused[2][:,1:], pts * config.scales[2], './outputs/%d_off_f2.jpg'%i)
				visutil.vis_offset(img, offset_map, pts, './outputs/%d_offgt2.jpg'%i)
				visutil.vis_offset(img, batch[1], pts * config.scales[0], './outputs/%d_offgt0.jpg'%i)
				visutil.vis_offset(img, batch[5], pts * config.scales[1], './outputs/%d_offgt1.jpg'%i)
				# print(outs.max(), outs.min(), hmap.max(), hmap.min(), mask.max(), mask.min())

			if i%20==0 and gpu==0:
				curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
				print('%s  Replica:%d  Progress:%d/%d  LsL:%.3e  LsM:%.3e  LsS:%.3e  LR:%.1e'%(curr_time, gpu, i, len(loader), ls_large, ls_medium, ls_small, lr))
				print('%s  Replica:%d  Progress:%d/%d  OfL:%.3e  OfM:%.3e  OfS:%.3e  LR:%.1e'%(curr_time, gpu, i, len(loader), of_large, of_medium, of_small, lr))
				print('%s  Replica:%d  Progress:%d/%d  LsLF:%.3e  LsMF:%.3e  LsSF:%.3e  LR:%.1e'%(curr_time, gpu, i, len(loader), ls_large_f, ls_medium_f, ls_small_f, lr))
				print('%s  Replica:%d  Progress:%d/%d  OfLF:%.3e  OfMF:%.3e  OfSF:%.3e  LR:%.1e'%(curr_time, gpu, i, len(loader), of_large_f, of_medium_f, of_small_f, lr))

		if e in config.lr_epoch:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr

		if e%config.save_interval==0 and gpu==0:
			stamp = random.randint(0, 1000000)
			saver.save('./model/%d_%d.pth'%(e, stamp))

if __name__=='__main__':
	main()
