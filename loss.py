import torch 
from TorchSUL import Model as M 
import torch.nn.functional as F 
import config 

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
		self.HM = HmapLoss()
		self.model = model 

	def run(self, img):
		all_fmaps = []
		all_hmaps = []
		for i in range(len(config.inp_scales)):
			scale = config.inp_scales[i]
			inp = F.interpolate(img, (scale, scale))
			o3, o2, o1, f3, f2, f1 = self.model.run(inp)
			all_fmaps.append([f3, f2, f1])
			all_hmaps.append([o3, o2, o1])
		return all_fmaps, all_hmaps

	def run_input(self, img):
		o3, o2, o1, f3, f2, f1 = self.model.run(img)
		return [f3, f2, f1], [o3, o2, o1]

	def forward(self, img, hmap, mask, debugmaps=None):
		'''
		Some explanation:
			We observe that training the network in low resolution (128) will help to produce more heatmaps for small
			persons thus reduce the miss detection. However, merely using small crops is not capable to handle large
			objects. Therefore, we need also to crop the larger areas into 128 and compute the heatmap again to produce
			multi-scale results. This is similar to typical multi-scale testing strategy but in a reverse manner (they 
			upsample the images but we downsample the images). This multi-scale & down-sampling process can effectively 
			help to produce more heatmaps especially for the small objects thus improve the recall rate.  

			Despite its effectiveness, the computation cost is quite much. Here we would like to use multi-scale training 
			by feeding the network with different resolutions and supervise	the heatmap under different down-sampling rates. 
			This process aims to incorporate the extrinsic "data manipulation" to intrinsic "multi-scale heatmaps". At the 
			same time, the network can "see" training data in different resolution, which could be a plus.
		'''
		mask = mask.float()
		all_hmaps = []  # [num_scales(3), BSize, num_pts(17), out_size(64), out_size(64)]
		all_losses = []  # [num_scales(3)]
		for i in range(len(config.inp_scales)):
			scale = config.inp_scales[i]
			inp = F.interpolate(img, (scale, scale))
			out_hmap = self.model(inp)[i]  # 512 -> 64, 256 -> 64, 128 -> 64
			# print(out_hmap.shape, hmap.shape, mask.unsqueeze(1).shape)
			hm = self.HM(out_hmap, hmap, mask.unsqueeze(1))
			all_hmaps.append(out_hmap)
			all_losses.append(hm)
		return all_losses, all_hmaps   # May add some extra outputs for the GCN part in the future 

class CircleLoss(M.Model):
	def initialize(self, gamma, m):
		self.gamma = gamma 
		self.m = m 
	def forward(self, x, pairwise):
		# pairwise: B*N*N matrix, 1 same, 0 ignore, -1 different 
		# x: B*N*F matrix
		x = x / torch.norm(x, 2, dim=-1, keepdim=True)
		sim = torch.einsum('ijk,ikl->ijl', x, x.transpose(-1, -2))
		alpha = - torch.clamp(pairwise * (self.m - sim) + torch.clamp(pairwise, 0), 0) * pairwise
		delta = 0.5 + pairwise * (0.5 - self.m)
		total = torch.exp(self.gamma * alpha * (sim - delta))  # B*N*N
		total = torch.sum(total, dim=(1,2))
		loss = torch.log(1 + total).mean()
		return loss 

class BiasLoss(M.Model):
	def forward(self, x, label):
		loss = torch.mean(torch.pow(x - label, 2))
		return loss 

class ConfLoss(M.Model):
	def forward(self, x, label):
		loss = F.binary_cross_entropy_with_logits(x, label)
		return loss 
