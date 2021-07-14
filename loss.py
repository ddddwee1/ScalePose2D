from os import altsep
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

	def forward_with_fmap(self, img, hmap, mask):
		mask = mask.float()
		all_fmaps = []
		all_hmaps = []  # [num_scales(3), BSize, num_pts(17), out_size(64), out_size(64)]
		all_losses = []  # [num_scales(3)]

		for i in range(len(config.inp_scales)):
			scale = config.inp_scales[i]
			inp = F.interpolate(img, (scale, scale))
			outs = self.model.run(inp)  # 512 -> 64, 256 -> 64, 128 -> 64
			o3, o2, o1, f3, f2, f1 = outs 
			out_hmap = outs[i]
			# print(out_hmap.shape, hmap.shape, mask.unsqueeze(1).shape)
			hm = self.HM(out_hmap, hmap, mask.unsqueeze(1))
			all_hmaps.append([o3, o2, o1])
			all_fmaps.append([f3, f2, f1])
			all_losses.append(hm)
		return all_losses, all_hmaps, all_fmaps   # May add some extra outputs for the GCN part in the future 

	def forward(self, img, hmap, mask):
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
		all_fmaps = []
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

	def compute_loss(self, x, pairwise):
		# pairwise: B*N*N matrix, 1 same, 0 ignore, -1 different 
		# x: B*N*F matrix
		x = x / torch.norm(x, 2, dim=-1, keepdim=True)
		sim = torch.einsum('ijk,ikl->ijl', x, x.transpose(-1, -2))
		# print('sim',sim.max(), sim.min())
		# alpha = - torch.clamp(pairwise * (self.m - sim) + torch.clamp(pairwise, 0), 0) * pairwise
		alpha = -pairwise   # may add scaling factor later, but use this for now 
		delta = torch.clamp(pairwise, 0) * self.m
		total = self.gamma * alpha * sim
		# print('total',total.max(), total.min(), self.gamma, alpha.max(), alpha.min())
		total = torch.exp(total)  # B*N*N
		
		total = torch.sum(total, dim=(1,2))
		loss = torch.log(1 + total).mean()
		return loss 

	def get_pariwise(self, inst, ignore):
		# inst: [bsize, out_scale, 17, k]
		# ignore: [bsize, out_scale, 17, k, 1]
		bsize = inst.shape[0]
		inst_flatten = inst.reshape(bsize, -1)
		pairwise = inst_flatten.unsqueeze(1) - inst_flatten.unsqueeze(2)
		pairwise = (pairwise==0).float() * 2 - 1
		mask_flatten = 1 - ignore.reshape(bsize, -1)
		pairwise = pairwise * mask_flatten.unsqueeze(1) * mask_flatten.unsqueeze(2)
		return pairwise

	def forward(self, x, inst, ignore):
		pairwise = self.get_pariwise(inst, ignore)
		bsize, n_scales = x.shape[0], x.shape[1]
		x = x.reshape(bsize, n_scales*config.num_pts*config.top_k_candidates, -1)
		loss = self.compute_loss(x, pairwise)
		return loss

class BiasLoss(M.Model):
	def forward(self, x, label, ignore):
		loss = torch.sum(torch.pow(x - label, 2) * (1 - ignore).unsqueeze(-1)) / torch.sum(1 - ignore) / config.out_size / config.out_size
		return loss 

class ConfLoss(M.Model):
	def forward(self, x, label, ignore):
		loss = F.binary_cross_entropy_with_logits(x, label, reduction='none') 
		loss = loss * (1 - ignore)
		loss = loss.mean()
		return loss 

class UnifiedNet(M.Model):
	def initialize(self, module, refine, sample_layer, label_generator):
		self.module = module
		self.refine = refine 
		self.sample_layer = sample_layer
		self.label_generator = label_generator
		self.circle = CircleLoss(16, 0.3)
		self.biasls = BiasLoss()
		self.confls = ConfLoss()

	def forward(self, x, hmap, mask, pts):
		hmap_losses, all_hmaps, all_fmaps = self.module.forward_with_fmap(x, hmap, mask)
		feat_losses = []
		bias_losses = []
		conf_losses = []
		for i in range(len(all_hmaps)):
			crops, centers, sizes = self.sample_layer(all_hmaps[i], all_fmaps[i], x)
			conf_label, bias_label, inst_label, ignore = self.label_generator(centers, sizes, pts, mask)
			# print('conf', conf_label.max(), conf_label.min(), ignore.max(), ignore.min())
			out_conf, out_bias, out_feat = self.refine(crops)
			loss_feat = self.circle(out_feat, inst_label, 1 - (1- ignore)*conf_label)  # only consider the joints: ignore=0 && conf=1
			loss_bias = self.biasls(out_bias, bias_label, ignore)
			loss_conf = self.confls(out_conf, conf_label, ignore)
			feat_losses.append(loss_feat)
			bias_losses.append(loss_bias)
			conf_losses.append(loss_conf)
		out_hmaps = [hms[i] for i,hms in enumerate(all_hmaps)]
		return hmap_losses, feat_losses, bias_losses, conf_losses, out_hmaps

