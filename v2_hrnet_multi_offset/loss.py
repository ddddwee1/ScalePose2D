import torch 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import torch.nn as nn 
import config 

class OffsetLoss(M.Model):
	def forward(self, pred, gt, weight, beta=1.0/9):
		l1 = torch.abs(pred - gt)
		cond = l1 < beta 
		loss = torch.where(cond, 0.5 * l1 ** 2 / beta, l1 - 0.5 * beta)
		weight_nums = (weight>0).float().sum()
		loss = torch.sum(loss * weight) / (weight_nums + 0.0001)
		return loss 

class HmapLoss(M.Model):
	def forward(self, hmap, gt, mask):
		# hmap = torch.sigmoid(hmap)
		loss = torch.pow(hmap - gt, 2)
		loss = loss * (1 - mask.expand_as(loss))
		# loss = loss * (gt.detach() * 10 + 1)
		loss = loss.mean()
		return loss 

class ModelWithLoss(M.Model):
	def initialize(self, model):
		self.Off = OffsetLoss()
		self.HM = HmapLoss()
		self.model = model 
	def forward(self, batch):
		img = batch[0]
		hmap = batch[-3]
		mask = batch[-2]
		offset_gt = batch[-5]
		offset_weight = batch[-4]

		all_losses_fused = []
		all_offls_fused = []
		all_maps_fused = []
		for i in range(len(config.inp_scales)):
			img_resized = F.interpolate(img, (config.inp_scales[i], config.inp_scales[i]))
			fused = self.model(img_resized)

			of_gt = batch[i*4+1]
			of_w = batch[i*4+2]
			hm_gt = batch[i*4+3]
			hm_msk = batch[i*4+4]
			hm_fused = self.HM(fused[:,0:1], hm_gt, hm_msk.unsqueeze(1))
			offls_fused = self.Off(fused[:,1:], of_gt, of_w)

			all_maps_fused.append(fused)
			all_losses_fused.append(hm_fused)
			all_offls_fused.append(offls_fused)

		return all_losses_fused, all_offls_fused, all_maps_fused

