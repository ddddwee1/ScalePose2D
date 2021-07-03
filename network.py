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
        # It works only when batch-norm
        self.c11 = M.ConvLayer(3, outchn, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
        self.c12 = M.ConvLayer(1, outchn*4, usebias=False)
        self.upsample = DepthToSpace(2)
        self.c21 = M.ConvLayer(3, outchn, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
        self.c22 = M.ConvLayer(1, config.num_pts)
    def forward(self, x):
        x = self.c11(x)
        x = self.c12(x)
        x = self.upsample(x)
        x = self.c21(x)
        x = self.c22(x)
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

class SamplingLayer(M.Model):
    def initialize(self):
        # Sample 
        self.nms = M.MaxPool2D(5)
    def forward(self, hmaps, fmaps):
        # 3 scales input and 3 scales output. All resize to config.out_size
        # TODO: debug it with some pre-pickled heatmap & feature maps 
        for inp_idx in range(3):
            for out_idx in range(3):
                hm = hmaps[inp_idx][out_idx]
                fm = fmaps[inp_idx][out_idx]
                hm_nms = self.nms(hm)
                
                bsize = hm.shape[0]
                hm_flat = hm.reshape(bsize, config.num_pts, -1)
                hm_nms_flat = hm_nms.reshape(bsize, config.num_pts, -1)
                hm_idx = torch.where(hm_flat==hm_nms_flat)

                print(len(hm_idx))

if __name__=='__main__':
    import torch 
    net = MultiScaleNet()
    x = torch.zeros(1, 3, 512, 512)
    o1, o2, o3 = net(x)
    print(o1.shape, o2.shape, o3.shape)
