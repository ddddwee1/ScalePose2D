import resnet
import TorchSUL.Model as M
import config
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision

class DepthToSpace(M.Model):
    def initialize(self, block_size):
        self.block_size = block_size

    def forward(self, x):
        bsize, chn, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        assert chn % (self.block_size**2) == 0, 'DepthToSpace: Channel must be divided by square(block_size)'
        x = x.view(bsize, -1, self.block_size, self.block_size, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape(bsize, -1, h*self.block_size, w*self.block_size)
        return x

class OutBlock(M.Model):
    def initialize(self, outchn):
        self.c11 = M.ConvLayer(3, outchn, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
        self.c12 = M.ConvLayer(1, outchn*4, usebias=False)
        self.upsample = DepthToSpace(2)
        self.c21 = M.ConvLayer(
            3, outchn, activation=M.PARAM_LRELU, batch_norm=True, usebias=False)
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
        # better use features from the second block, otherwise it might be too shallow for high-resolution hmaps
        self.out1 = OutBlock(64)
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
        return result

    def get_joint_candidates(self, hmap_list, k=40):
        roi_candidates = []  # [out_scales, bsize, 17, k, 2]
        candidate_sizes = []

        for j in range(len(config.inp_scales)):  # feature map scales
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
        return roi_candidates, candidate_sizes, topk_val

    def forward(self, hmap_list, fmap_list, img):
        # The inp_scale should be emitted here, because in testing we use only one scale
        # Make the GCN consistent between training and testing. The input scale should be randomly sampled from possible values
        k = config.top_k_candidates
        bsize = hmap_list[0].shape[0]
        roi_candidates, candidate_sizes, _ = self.get_joint_candidates(hmap_list, k=k)

        all_fhmap = []   # [out_scales, bsize, 17*k, channel_sum]
        for coord, size in zip(roi_candidates, candidate_sizes):
            scale_fhmap = []
            for fmap, hmap in zip(fmap_list, hmap_list):
                w = fmap.shape[3]
                chn = fmap.shape[1]
                # [bsize, 17, k, 4]
                box = self._get_bbox(coord, config.roi_box_size, size, w)
                box = box.reshape(box.shape[0], -1, 4)
                fmap_cropped = torchvision.ops.roi_align(fmap, list(box), int(2048 * 3 // chn))  # [bsize*17*k, chn, 7, 7]
                fmap_cropped = fmap_cropped.reshape(bsize, config.num_pts*k, -1)  # [bsize, 17*k, chn*7*7]

                w = hmap.shape[3]
                hmap_cropped = torchvision.ops.roi_align(hmap, list(box), 7)  # [bsize*17*k, chn, 7, 7]
                hmap_cropped = hmap_cropped.reshape(bsize, config.num_pts*k, -1)  # [bsize, 17*k, chn*7*7]
                fhmap = torch.cat([fmap_cropped, hmap_cropped], dim=-1)
                scale_fhmap.append(fhmap)

            w = img.shape[3]
            box = self._get_bbox(coord, config.roi_box_size, size, w)
            box = box.reshape(box.shape[0], -1, 4)
            img_cropped = torchvision.ops.roi_align(img, list(box), 13)
            img_cropped = img_cropped.reshape(bsize, config.num_pts*k, -1)  # [bsize, 17*k, chn*13*13]
            scale_fhmap.append(img_cropped)

            scale_fhmap = torch.cat(scale_fhmap, dim=-1)
            print(scale_fhmap.shape)
            all_fhmap.append(scale_fhmap)

        all_fhmap = torch.stack(all_fhmap, dim=1)  # [bsize, out_scales, 17*k, chn_sum]
        all_fhmap = all_fhmap.reshape(bsize, all_fhmap.shape[1], config.num_pts, k, all_fhmap.shape[-1])  # [bsize, out_scales, 17, k, chn_sum]
        roi_candidates = torch.stack(roi_candidates, dim=1)  # [bsize, out_scales, 17, k, 2]
        # candidate_sizes = torch.stack(candidate_sizes)  # [out_scales]
        return all_fhmap, roi_candidates, candidate_sizes

# bsize*17*k can be the data length, apply attention on this dimension

class LabelProducer(M.Model):
    def forward(self, roi_candidates, candidate_sizes, gt_pts, mask):
        # produce 3 labels: confidence, bias, ID tag; and flag: ignore (if the joint locates on unlabelled instances)
        # roi_candidates: [bsize, out_scales, 17, k, 2]
        # candidate_sizes: [out_scales]
        # gt_pts: [bsize, max_inst, 17, 3]
        # mask: [bsize, out_size, out_size]

        gt_pts = gt_pts.permute([0, 2, 1, 3])
        pts_xy = gt_pts[:, :, :, :2]   # xy [bsize, 17, inst, 2]
        pts_vis = gt_pts[:, :, :, 2:3]  # vis: [bsize, 17, inst, 1]; 0 for invisible, 1 for visible

        # get confidence
        scales = config.out_size / candidate_sizes
        scales = scales.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        roi_candidates = roi_candidates * scales
        left_upper = roi_candidates - config.roi_box_size * scales
        right_lower = roi_candidates + config.roi_box_size * scales        # [bsize, out_scales, 17, k, 1, 2]
        left_upper = left_upper.unsqueeze(4)
        right_lower = right_lower.unsqueeze(4)
        pts_xy = pts_xy.unsqueeze(1).unsqueeze(3)  # [bsize, 1, 17, 1, inst, 2]
        pts_vis = pts_vis.unsqueeze(1).unsqueeze(3)  # [bsize, 1, 17, 1, inst, 1]
        conf, _ = torch.max(torch.prod((pts_xy < right_lower).float() * (pts_xy > left_upper).float() * pts_vis.float(), dim=-1), -1)  # [bsize, out_scales, 17, k]

        # get id_tag
        bias = pts_xy - roi_candidates.unsqueeze(4)  # [bsize, out_scales, 17, k, inst, 2]
        dist = torch.sum(torch.pow(bias, 2), -1)  # [bsize, out_scales, 17, k, inst]
        _, nearest_inst = torch.min(dist, -1)  # [bsize, out_scales, 17, k]

        # get bias
        inst = nearest_inst.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, -1, -1, 2)
        bias = torch.gather(bias, 4, inst).suqeeze(4)  # [bsize, out_scales, 17, k, 2]

        # get ignore
        bsize = roi_candidates.shape[0]
        mask_flatten = mask.reshape(bsize, -1)  
        roi_candidates_flatten = roi_candidates[:, :, :, :, 0] + roi_candidates[:, :, :, :, 1] * config.out_size  # [bsize, out_scales, 17, k]
        roi_candidates_flatten = roi_candidates_flatten.reshape(bsize, -1)  # [bsize, -1]
        ignore = torch.gather(mask_flatten, 1, roi_candidates_flatten.int())  # [bsize, out_scales * 17 * k]
        ignore = ignore.reshape(bsize, -1, config.num_pts, config.top_k_candidates)  # [bsize, out_scales, 17, k]
        return conf, bias, nearest_inst, ignore

class Attention(M.Model):
    def initialize(self, dim, num_heads, attn_drop):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop = attn_drop

        self.qkv = M.Dense(dim*3, usebias=True)
        self.proj = M.Dense(dim)

    def forward(self, x, return_attn=False):
        # print(x.shape)
        qkv = self.qkv(x)
        B, N, _ = qkv.shape
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, n_heads, N, head_dim

        attn = (q @ k.transpose(-1, -2)) * self.scale    # B, n_heads, N, N
        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, self.attn_drop, self.training, False)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.head_dim * self.num_heads)
        x = self.proj(x)
        if return_attn:
            return x, attn
        return x

class MLP(M.Model):
    def initialize(self, dim, mlp_ratio):
        self.fc1 = M.Dense(dim*mlp_ratio, usebias=True, activation=M.PARAM_GELU)
        self.fc2 = M.Dense(dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class AttentionBlock(M.Model):
    def initialize(self, dim, num_heads, mlp_ratio=4, attn_drop=0.0, drop_path=0.0):
        self.drop_path = drop_path
        self.norm1 = M.LayerNorm(1)
        self.attn = Attention(dim, num_heads, attn_drop)
        self.norm2 = M.LayerNorm(1)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x1 = self.norm1(x)
        x1 = self.attn(x1)
        x1 = F.dropout(x1, self.drop_path, self.training, False)
        x = x1 + x

        x1 = self.norm2(x)
        x1 = self.mlp(x1)
        x1 = F.dropout(x1, self.drop_path, self.training, False)
        x = x1 + x
        return x

    def forward_attn(self, x):
        x1 = self.norm1(x)
        x1, att = self.attn(x1, return_attn=True)
        x1 = F.dropout(x1, self.drop_path, self.training, False)
        x = x1 + x

        x1 = self.norm2(x)
        x1 = self.mlp(x1)
        x1 = F.dropout(x1, self.drop_path, self.training, False)
        x = x1 + x
        return x, att

class PosEmbed(M.Model):
    def build_forward(self, x):
        B, N, K, topk, chn = x.shape
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, K, 1, 1024))

        nn.init.uniform_(self.pos_emb, -0.2, 0.2)
        pos_emb = self.pos_emb.expand(B, N, -1, topk, -1)
        return torch.cat([x, pos_emb], dim=-1)

    def forward(self, x):
        B, N, K, topk, chn = x.shape
        pos_emb = self.pos_emb.expand(B, N, -1, topk, -1)
        return torch.cat([x, pos_emb], dim=-1)

class RefineNet(M.Model):
    def initialize(self, feat_dim, num_heads, pos_embed, depth, attn_drop=0.0, drop_path=0.0):
        if pos_embed:
            self.pos_embed = PosEmbed()
        else:
            self.pos_embed = None
        self.fc1 = M.Dense(feat_dim)
        self.attn_blocks = nn.ModuleList()
        for i in range(depth):
            self.attn_blocks.append(AttentionBlock(feat_dim, num_heads, attn_drop, drop_path))

        self.fc2 = M.Dense(feat_dim)
        self.out_conf = M.Dense(1)
        self.out_bias = M.Dense(2)
        self.out_feat = M.Dense(128)

    def forward(self, x):
        # x: [bsize, out_scales, 17, k, num_chn]
        bsize, n_scales = x.shape[0], x.shape[1]
        if self.pos_embed:
            x = self.pos_embed(x)

        x = x.reshape(bsize, n_scales*config.num_pts * config.top_k_candidates, -1)
        for blk in self.attn_blocks:
            x = blk(x)
        x = self.fc2(x)
        conf = self.out_conf(x)
        bias = self.out_bias(x)
        feat = self.out_feat(x)

        conf = conf.reshape(bsize, n_scales, config.num_pts, config.top_k_candidates)
        bias = bias.reshape(bsize, n_scales, config.num_pts, config.top_k_candidates, 2)
        feat = feat.reshape(bsize, n_scales, config.num_pts, config.top_k_candidates, 128)
        return conf, bias, feat

if __name__ == '__main__':
    import torch
    net = MultiScaleNet()
    x = torch.zeros(1, 3, 512, 512)
    o1, o2, o3 = net(x)
    print(o1.shape, o2.shape, o3.shape)
