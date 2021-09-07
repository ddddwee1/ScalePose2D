import resnet
import torch
import TorchSUL.Model as M 
import torch.nn.functional as F 
import config 

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

class OutBlock(M.Model):
    def initialize(self, outchn, out_pts):
        self.c11 = M.ConvLayer(3, outchn, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
        self.c12 = M.ConvLayer(1, outchn*4, usebias=False)
        self.upsample = DepthToSpace(2)
        self.c21 = M.ConvLayer(3, outchn, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
        self.c22 = M.ConvLayer(1, out_pts)
    def forward(self, x):
        x = self.c11(x)
        x = self.c12(x)
        x = self.upsample(x)
        pre_x = x = self.c21(x)
        x = self.c22(x)
        return x, pre_x

class UpBlock(M.Model):
    def initialize(self, outchn):
        self.c11 = M.ConvLayer(3, outchn, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
        self.c12 = M.ConvLayer(1, outchn*4, usebias=False)
        self.upsample = DepthToSpace(2)
        self.c21 = M.ConvLayer(3, outchn, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
    def forward(self, x):
        x = self.c11(x)
        x = self.c12(x)
        x = self.upsample(x)
        x = self.c21(x)
        return x 

class HmapPts(M.Model):
    def initialize(self, outchn):
        self.b1 = OutBlock(outchn, 1)
        self.b2 = OutBlock(outchn, config.num_pts*2)

    def forward(self, x):
        o1, pre_o1 = self.b1(x)
        o2, pre_o2 = self.b2(x)
        return torch.cat([o1, o2], dim=1), torch.cat([pre_o1, pre_o2], dim=1)

class Fusion(M.Model):
    def initialize(self):
        self.up1 = UpBlock(64)
        self.c1 = M.ConvLayer(3, 64, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
        self.c2 = M.ConvLayer(3, 64, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
        self.c3 = M.ConvLayer(1, (1+config.num_pts*2)*1)
    
    def forward(self, o1, o2, o3, f1, f2, f3, shallow_feat):
        f3 = F.interpolate(f3, None, scale_factor=4)
        f2 = F.interpolate(f2, None, scale_factor=2)
        f = torch.cat([f1, f2, f3], dim=1)
        x = self.up1(f)

        o3 = F.interpolate(o3, None, scale_factor=4)
        o2 = F.interpolate(o2, None, scale_factor=2)
        x = torch.cat([shallow_feat, x, o1, o2, o3], dim=1)
        
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        # x = x.reshape(x.shape[0], 3, 1+config.num_pts*2, x.shape[-2], x.shape[-1])  #[bsize, 3, 1+2k, h, w]
        return x 

class MultiScaleNet(M.Model):
    def initialize(self):
        self.backbone = resnet.Res50()
        self.out1 = HmapPts(64)  # better use features from the second block, otherwise it might be too shallow for high-resolution hmaps 
        self.out2 = HmapPts(64)
        self.out3 = HmapPts(64)
        self.fuse = Fusion()
    
    def forward(self, x):
        x, fmap0, fmap1, fmap2, fmap3 = self.backbone(x)
        o1, pre_o1 = self.out1(fmap1)
        o2, pre_o2 = self.out2(fmap2)
        o3, pre_o3 = self.out3(fmap3)
        fused = self.fuse(pre_o1, pre_o2, pre_o3, fmap1, fmap2, fmap3, shallow_feat=fmap0)
        # return o3, o2, o1, torch.stack([o1, o1, o1], dim=1)
        return o3, o2, o1, fused

if __name__=='__main__':
    import torch 
    net = MultiScaleNet()
    x = torch.zeros(1, 3, 512, 512)
    o1, o2, o3 = net(x)
    print(o1.shape, o2.shape, o3.shape)
