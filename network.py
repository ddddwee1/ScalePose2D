import resnet
import TorchSUL.Model as M 
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
    def initialize(self, outchn):
        self.c1 = M.ConvLayer(3, outchn*4, activation=M.PARAM_RELU)
        self.upsample = DepthToSpace(2)
        self.c2 = M.ConvLayer(3, config.num_pts)
    def forward(self, x):
        x = self.c1(x)
        x = self.upsample(x)
        x = self.c2(x)
        return x 

class MultiScaleNet(M.Model):
    def initialize(self):
        self.backbone = resnet.Res50()
        self.out1 = OutBlock(64)  # better use features from the second block, otherwise it might be too shallow for high-resolution hmaps 
        self.out2 = OutBlock(64)
        self.out3 = OutBlock(64)
    
    def forward(self, x):
        x, fmap1, fmap2, fmap3 = self.backbone(x)
        o1 = self.out1(fmap1)
        o2 = self.out2(fmap2)
        o3 = self.out3(fmap3)
        return o3, o2, o1
        
if __name__=='__main__':
    import torch 
    net = MultiScaleNet()
    x = torch.zeros(1, 3, 512, 512)
    o1, o2, o3 = net(x)
    print(o1.shape, o2.shape, o3.shape)
