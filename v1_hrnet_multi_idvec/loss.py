import torch 
import torch.nn.functional as F 
from TorchSUL import Model as M 
import torch.nn as nn 
import config 

class AELoss(M.Model):
	def inst_ae(self, idmap, pts):
		area = 0
		pull_loss = 0

		conf = pts[:,:,2]
		pts = pts[:,:,:2] * idmap.shape[2] / config.out_size
		pts = pts.long()
		tags_means = []
		for i in range(config.max_inst):
			tags = []
			for j in range(config.num_pts):
				if conf[i,j]>0:
					x,y = pts[i,j,0], pts[i,j,1]
					for xx in range(x-area, x+area+1):
						for yy in range(y-area, y+area+1):
							if xx<0 or xx>=idmap.shape[2] or yy<0 or yy>=idmap.shape[2]:
								continue 
							tag = idmap[:,yy,xx]
							tags.append(tag)
			if len(tags)==0:
				continue
			tags = torch.stack(tags, dim=0)
			tags_mean = tags.mean(dim=0)
			intra = torch.pow(tags - tags_mean, 2).mean()
			pull_loss = pull_loss + intra
			tags_means.append(tags_mean)

		num_tags = len(tags_means)

		if num_tags==0:
			return torch.zeros(1)[0].float().to(idmap.device), torch.zeros(1)[0].float().to(idmap.device)
		
		pull_loss = pull_loss / num_tags
		if num_tags==1:
			return torch.zeros(1)[0].float().to(idmap.device), pull_loss

		# tags_means = torch.stack(tags_means)
		# A = tags_means.unsqueeze(0)
		# B = A.permute(0,1)
		# diff = torch.triu(A-B, diagonal=1)
		# push = torch.sum(torch.exp(- torch.pow(diff, 2)))
		# push_loss = push / ((num_tags-1) * num_tags)

		# A = tags_means.expand(num_tags, num_tags)
		# B = A.permute(0, 1)
		# diff = torch.pow(A-B, 2)
		# push = torch.sum(torch.exp(-diff)) - num_tags
		# push_loss = push * 0.5 / ((num_tags-1) * num_tags)

		push_loss = 0
		for i in range(num_tags):
			for j in range(num_tags):
				if i!=j:
					diff = torch.pow(tags_means[i] - tags_means[j], 2).sum()
					diff = torch.exp(-diff / config.feat_dim)
					push_loss = push_loss + diff
		push_loss = push_loss * 0.5 / ((num_tags-1) * num_tags)

		return push_loss, pull_loss

	def forward(self, idmap, pts):
		bsize = idmap.shape[0]
		push_loss = []
		pull_loss = []
		for i in range(bsize):
			push, pull = self.inst_ae(idmap[i], pts[i])
			push_loss.append(push)
			pull_loss.append(pull)
		push_loss = torch.stack(push_loss)
		pull_loss = torch.stack(pull_loss)
		return push_loss, pull_loss

class HmapLoss(M.Model):
	def forward(self, hmap, gt, mask):
		# hmap = torch.sigmoid(hmap)
		loss = torch.pow(hmap - gt, 2)
		loss = loss * (1 - mask.expand_as(loss))
		# loss = loss * (gt.detach() * 10 + 1)
		loss = loss.mean(dim=3).mean(dim=2).mean(dim=1)
		return loss 

class ModelWithLoss(M.Model):
	def initialize(self, model):
		self.AE = AELoss()
		# self.HM = HmapLoss()
		self.model = model 
	def forward(self, img, hmap, mask, pts):
		all_pull = []
		all_push = []
		all_idout = []
		for i in range(len(config.inp_scales)):
			img_resized = F.interpolate(img, (config.inp_scales[i], config.inp_scales[i]))
			idout = self.model(img_resized)
			push, pull = self.AE(idout, pts)
			all_push.append(push)
			all_pull.append(pull)
			all_idout.append(idout)
		return all_push, all_pull, all_idout

