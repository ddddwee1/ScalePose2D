import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from TorchSUL import Model as M 
import config 
import hrnet 
from torch.nn.parallel import replicate, scatter, parallel_apply, gather

class ResUnit(M.Model):
	def initialize(self, out, stride, shortcut=False):
		self.shortcut = shortcut
		self.c1 = M.ConvLayer(1, out//4, usebias=False, activation=M.PARAM_RELU, batch_norm=True)
		self.c2 = M.ConvLayer(3, out//4, usebias=False, activation=M.PARAM_RELU, pad='SAME_LEFT', stride=stride, batch_norm=True)
		self.c3 = M.ConvLayer(1, out, usebias=False, batch_norm=True)
		if shortcut:
			self.sc = M.ConvLayer(1, out, usebias=False, stride=stride, batch_norm=True)

	def forward(self, x):
		branch = self.c1(x)
		branch = self.c2(branch)
		branch = self.c3(branch)
		if self.shortcut:
			sc = self.sc(x)
		else:
			sc = x 
		res = branch + sc
		res = M.activation(res, M.PARAM_RELU)
		return res 

class Head(M.Model):
	def initialize(self, head_layernum, head_chn):
		self.layers = nn.ModuleList()
		for i in range(head_layernum):
			self.layers.append(M.ConvLayer(3, head_chn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
	def forward(self, x):
		for l in self.layers:
			x = l(x)
		return x 

class DepthToSpace(M.Model):
	def initialize(self, block_size):
		self.block_size = block_size
	def forward(self, x):
		bsize, chn, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		assert chn%(self.block_size**2)==0, 'DepthToSpace: Channel must be divided by square(block_size)'
		x = x.view(bsize, -1, self.block_size, self.block_size, h, w)
		x = x.permute(0,1,4,2,5,3)
		x = x.reshape(bsize, -1, h*self.block_size, w*self.block_size)
		return x 

class UpSample(M.Model):
	def initialize(self, upsample_layers, upsample_chn):
		self.prevlayers = nn.ModuleList()
		#self.uplayer = M.DeConvLayer(3, upsample_chn, stride=2, activation=M.PARAM_PRELU, batch_norm=True, usebias=False)
		self.uplayer = M.ConvLayer(3, upsample_chn*4, activation=M.PARAM_PRELU, usebias=False)
		self.d2s = DepthToSpace(2)
		self.postlayers = nn.ModuleList()
		for i in range(upsample_layers):
			self.prevlayers.append(M.ConvLayer(3, upsample_chn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
		for i in range(upsample_layers):
			self.postlayers.append(M.ConvLayer(3, upsample_chn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False))
	def forward(self, x):
		for p in self.prevlayers:
			x = p(x)
		x = self.uplayer(x)
		x = self.d2s(x)
		# print('UPUP', x.shape)
		for p in self.postlayers:
			x = p(x)
		return x 

class SingleJoint(M.Model):
	def initialize(self, chn, outchn):
		self.c1 = ResUnit(chn, 1)
		self.c2 = ResUnit(chn, 1)
		self.c3 = M.ConvLayer(1, outchn)

	def build_forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		nn.init.normal_(self.c3.conv.weight, std=0.001)
		return x 

	def forward(self, x):
		x = self.c1(x)
		x = self.c2(x)
		x = self.c3(x)
		return x 

class SeparateConv(M.Model):
	def initialize(self, num_pts):
		self.num_pts = num_pts
		self.branches = nn.ModuleList()
		for i in range(num_pts):
			self.branches.append(SingleJoint(16, 2))
	def forward(self, x):
		outs = []
		for i in range(self.num_pts):
			inp = x[:, i*16:i*16+16]
			out = self.branches[i](inp)
			outs.append(out)
		out = torch.cat(outs, dim=1)
		return out 


class MultiScaleNet(M.Model):
	def initialize(self):
		self.backbone = hrnet.Body()
		self.transition = M.ConvLayer(3, config.num_pts * 16, activation=M.PARAM_PRELU, batch_norm=True, usebias=False)
		self.transition_hm = M.ConvLayer(3, 32, activation=M.PARAM_PRELU, batch_norm=True, usebias=False)
		self.offset_conv = SeparateConv(config.num_pts)
		self.hm_conv = SingleJoint(32, config.num_pts+1)

	def forward(self, x):
		feat = self.backbone(x)
		feat_off = self.transition(feat)
		feat_hm = self.transition_hm(feat)
		off = self.offset_conv(feat_off)
		hm = self.hm_conv(feat_hm)
		return hm, off
