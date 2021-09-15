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
		loss = loss * (1 - mask)
		# loss = loss * (gt.detach() * 10 + 1)
		loss = loss.mean()
		return loss 

class ModelWithLoss(M.Model):
	def initialize(self, model):
		self.Off = OffsetLoss()
		self.HM = HmapLoss()
		self.model = model 
	def forward(self, img, heatmap, mask, offset, offset_weight):
		all_losses_fused = []
		all_offls_fused = []
		all_maps_fused = []

		hm, off = self.model(img)

		hm_loss = self.HM(hm, heatmap, mask.unsqueeze(1))
		off_loss = self.Off(off, offset, offset_weight)

		return hm, off, hm_loss, off_loss

