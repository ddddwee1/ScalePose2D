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

	model_dnet = network.DensityNet(config.head_layernum, config.head_chn, config.upsample_layers, config.upsample_chn)
	x = np.float32(np.random.random(size=[1,3,512,512]))
	x = torch.from_numpy(x)
	with torch.no_grad():
		model_dnet(x)

	M.Saver(model_dnet.backbone).restore('./model_imagenet_w32/')
	model = loss.ModelWithLoss(model_dnet)
	saver = M.Saver(model)
	saver.restore('./model/')

	torch.cuda.set_device(gpu)
	model.cuda(gpu)
	model.train()
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
	print('Model get.')

	loader, sampler = datareader.get_train_dataloader(16)
	optim = torch.optim.Adam(model.parameters(), lr=config.init_lr)

	for e in range(config.max_epoch):
		print('Replica:%d Epoch:%d'%(gpu, e))
		sampler.set_epoch(e)
		for i, (img, hmap, mask, pts) in enumerate(loader):
			# print(img.shape)
			optim.zero_grad()
			push_losses, pull_losses, idouts = model(img, hmap, mask, pts)

			push_large = push_losses[0].mean()
			push_med = push_losses[1].mean()
			push_small = push_losses[2].mean()

			pull_large = pull_losses[0].mean()
			pull_med = pull_losses[1].mean()
			pull_small = pull_losses[2].mean()

			push_loss = push_large + push_med + push_small
			pull_loss = pull_large + pull_med + pull_small
			if e<50:
				loss_total = 0.001*push_loss + 0.0001 * pull_loss
			else:
				loss_total = 0.001*push_loss + 0.001*pull_loss
			loss_total.backward()
			optim.step()
			lr = optim.param_groups[0]['lr']

			if i%100==0 and gpu==0:
				visutil.vis_batch(img, idouts[0], './outputs/%d_id_L.jpg'%i, minmax=True)
				visutil.vis_batch(img, idouts[1], './outputs/%d_id_M.jpg'%i, minmax=True)
				visutil.vis_batch(img, idouts[2], './outputs/%d_id_S.jpg'%i, minmax=True)

			if i%20==0:
				if gpu==0:
					curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
					print('%s  Replica:%d  Progress:%d/%d  IDsL:%.3e  IDsM:%.3e  IDsS:%.3e  LR:%.1e'%(curr_time, gpu, i, len(loader), pull_large, pull_med, pull_small, lr))
					print('%s  Replica:%d  Progress:%d/%d  IDdL:%.3e  IDdM:%.3e  IDdS:%.3e  LR:%.1e'%(curr_time, gpu, i, len(loader), push_large, push_med, push_small, lr))

		if e in config.lr_epoch:
			newlr = lr * 0.1 
			for param_group in optim.param_groups:
				param_group['lr'] = newlr

		if e%config.save_interval==0 and gpu==0:
			stamp = random.randint(0, 1000000)
			saver.save('./model/%d_%d.pth'%(e, stamp))

if __name__=='__main__':
	main()
