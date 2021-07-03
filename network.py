import resnet
import TorchSUL.Model as M 
import config 
import torch 
import torchvision

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

    def run(self, x):
        x, fmap1, fmap2, fmap3 = self.backbone(x)
        o1 = self.out1(fmap1)
        o2 = self.out2(fmap2)
        o3 = self.out3(fmap3)
        return o3, o2, o1, fmap3, fmap2, fmap1

class SamplingLayer(M.Model):
    def initialize(self):
        self.nms = M.MaxPool2D(5)

    def _get_bbox(self, center, box_size, source_scale, target_scale):
        # given a box_size on the source_scale, and map to the box in the target scale
        top_left = (center - box_size) * target_scale / source_scale
        btm_right = (center + box_size) * target_scale / source_scale
        result = torch.cat([top_left, btm_right], dim=-1)
        # print(result.shape)
        return result

    def forward(self, hmap_list, fmap_list, img, k=40):
        # The inp_scale should be emitted here, because in testing we use only one scale
        # Make the GCN consistent between training and testing. The input scale should be randomly sampled from possible values
        # expected output : [BSize, out_scales, 17, k, sum_feature_channels, ROIh, ROIw], [BSize, out_scales 17, k, h, w]
        roi_candidates = [] # [out_scales, bsize, 17, k, 2]
        candidate_sizes = []

        for j in range(len(config.inp_scales)): # feature map scales 
            hm = hmap_list[j]
            bsize = hm.shape[0]
            h = hm.shape[2]
            w = hm.shape[3]

            # get the coordinates of topk points 
            hm_nms = self.nms(hm)
            hm_filtered = (hm == hm_nms).float() * hm
            hm_filtered = hm_filtered.reshape(bsize, config.num_pts, h*w)
            topk_val, topk_idx = torch.topk(hm_filtered, k, dim=2)
            topk_w = torch.remainder(topk_idx, w)
            topk_h = torch.div(topk_idx, w)

            # store the index for inferring the roi area 
            roi_candidates.append(torch.stack([topk_w, topk_h], dim=-1))
            candidate_sizes.append(w)
        
        all_fhmap = []   # [out_scales, bsize, 17*k, channel_sum]
        for coord, size in zip(roi_candidates, candidate_sizes):
            scale_fhmap = []
            for fmap, hmap in zip(fmap_list, hmap_list):
                w = fmap.shape[3]
                box = self._get_bbox(coord, 2, size, w) # [bsize, 17, k, 4]
                box = box.reshape(box.shape[0], -1, 4)
                fmap_cropped = torchvision.ops.roi_align(fmap, list(box), min(12, max(4,8*w//size))) # [bsize*17*k, chn, 7, 7]
                fmap_cropped = fmap_cropped.reshape(bsize, config.num_pts*k, -1)  # [bsize, 17*k, chn*7*7]

                w = hmap.shape[3]
                box = self._get_bbox(coord, 2, size, w) # [bsize, 17, k, 4]
                box = box.reshape(box.shape[0], -1, 4)
                hmap_cropped = torchvision.ops.roi_align(hmap, list(box), 7) # [bsize*17*k, chn, 7, 7]
                hmap_cropped = hmap_cropped.reshape(bsize, config.num_pts*k, -1) #[bsize, 17*k, chn*7*7]
                print(fmap_cropped.shape, hmap_cropped.shape)
                fhmap = torch.cat([fmap_cropped, hmap_cropped], dim=-1)
                scale_fhmap.append(fhmap)  
            scale_fhmap = torch.cat(scale_fhmap, dim=-1)
            all_fhmap.append(scale_fhmap)
        
# TODO: Infer the labels 
# TODO: bsize*17*k can be the data length, apply attention on this dimension 

if __name__=='__main__':
    import torch 
    net = MultiScaleNet()
    x = torch.zeros(1, 3, 512, 512)
    o1, o2, o3 = net(x)
    print(o1.shape, o2.shape, o3.shape)
