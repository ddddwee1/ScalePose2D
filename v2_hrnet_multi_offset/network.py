import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from TorchSUL import Model as M 
import config 
import hrnet 
from torch.nn.parallel import replicate, scatter, parallel_apply, gather

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

class MultiScaleNet(M.Model):
	def initialize(self):
		self.backbone = hrnet.Body()
		self.upsample = UpSample(1, 32)
		self.head2 = Head(1, 32)
		self.c2 = M.ConvLayer(1, 1+ 2*config.num_pts)

	def build_forward(self, x, *args, **kwargs):
		feat = self.backbone(x)
		feat = self.upsample(feat)
		feat2 = self.head2(feat)
		idout = self.c2(feat2)
		nn.init.normal_(self.c2.conv.weight, std=0.001)
		print('normal init for last conv ')
		return idout
		
	def forward(self, x, density_only=False):
		feat = self.backbone(x)
		feat = self.upsample(feat)
		h2 = self.head2(feat)
		idout = self.c2(h2)
		return idout
