import cv2
import lightning
import math
import matplotlib
matplotlib.use('agg')
from matplotlib import cm
import numpy as np
import os
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from ocr.utils.shape import get_sample_point


class Block(nn.Module):
    def __init__(self, inplanes, planes, dcn = False):
        super(Block, self).__init__()
        self.dcn = dcn
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.resid = None
        if inplanes != planes:
            self.resid = nn.Conv2d(inplanes, planes, 1, 1, 0, bias = False)

    def forward(self, x):
        residual = x.clone()
        if self.resid:
            residual = self.resid(residual)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)

        return x


class FSNet(nn.Module):
    def __init__(self, channels = 64, numofblocks = 4, layers = [1,2,3,4], dcn = False):
        super(FSNet, self).__init__()
        self.channels = channels
        self.numofblocks = numofblocks
        self.layers = layers
        self.blocks = nn.ModuleList()
        self.steps = nn.ModuleList()

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 7, 2, 3, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        for l in layers:
            self.steps.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, 2, 1, bias = False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(True),
                )
            )
            next_channels = self.channels * l
            for i in range(l):
                tmp = [Block(channels, next_channels, dcn = False)]
                for j in range(self.numofblocks-1):
                    tmp.append(Block(next_channels, next_channels, dcn = dcn))
                self.blocks.append(nn.Sequential(*tmp))
            channels = next_channels

    def forward(self, x):
        x = self.stem(x)
        x1 = self.steps[0](x)

        x1 = self.blocks[0](x1)
        x2 = self.steps[1](x1)

        x1 = self.blocks[1](x1)
        x2 = self.blocks[2](x2)
        x3 = self.steps[2](x2)
        x1,x2 = switchLayer(self.channels, [x1,x2])

        x1 = self.blocks[3](x1)
        x2 = self.blocks[4](x2)
        x3 = self.blocks[5](x3)
        x4 = self.steps[3](x3)
        x1,x2,x3 = switchLayer(self.channels, [x1,x2,x3])

        x1 = self.blocks[6](x1)
        x2 = self.blocks[7](x2)
        x3 = self.blocks[8](x3)
        x4 = self.blocks[9](x4)

        return x1,x2,x3,x4


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def FSNet_M(pretrained = True):
    model = FSNet()
    print("MixNet backbone parameter size: ", count_parameters(model))
    if pretrained:
        load_path = "./pretrained/triHRnet_Synth_weight.pth"
        cpt = torch.load(load_path)
        model.load_state_dict(cpt, strict=True)
        print("load pretrain weight from {}. ".format(load_path))
        # print("mixHRnet does not have pretrained weight yet. ")
    return model


class BasisBlock(nn.Module):
    def __init__(self, inplanes, planes, groups = 1):
        super(BasisBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.resid = None
        if inplanes != planes:
            self.resid = nn.Conv2d(inplanes, planes, 1, 1, 0, bias = False)

    def forward(self, x):
        residual = x.clone()
        if self.resid:
            residual = self.resid(residual)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)

        return x


def switchLayer(channels, xs):
    numofeature = len(xs)
    splitxs = []
    for i in range(numofeature):
        splitxs.append(
            list(torch.chunk(xs[i], numofeature, dim = 1))
        )
    
    for i in range(numofeature):
        h,w = splitxs[i][i].shape[2:]
        tmp = []
        for j in range(numofeature):
            if i > j:
                splitxs[j][i] = F.avg_pool2d(splitxs[j][i], kernel_size = (2**(i-j)))
            elif i < j: 
                # splitxs[j][i] = F.interpolate(splitxs[j][i], (h,w), mode = 'bilinear')
                splitxs[j][i] = F.interpolate(splitxs[j][i], (h,w))
            tmp.append(splitxs[j][i])
        xs[i] = torch.cat(tmp, dim = 1)

    return xs


class FeatureShuffleNet(nn.Module):
    def __init__(self, block, channels = 64, numofblocks = None, groups = 1):
        super(FeatureShuffleNet, self).__init__()
        self.channels = channels
        self.numofblocks = numofblocks
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 7, 2, 3, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias = False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
        )

        Layerplanes = [self.channels, self.channels, self.channels*2, self.channels*3, self.channels*4]
        self.downSteps = nn.ModuleList()
        for planes in Layerplanes[:-1]:
            self.downSteps.append(
                nn.Sequential(
                    nn.Conv2d(planes, planes, 3, 2, 1, bias = False),
                    nn.BatchNorm2d(planes),
                    nn.ReLU(True),
                )
            )

        self.blocks_1 = nn.ModuleList()
        self.blocks_2 = nn.ModuleList()
        self.blocks_3 = nn.ModuleList()


        for l in range(4):
            for i, num in enumerate(self.numofblocks[l]):
                tmp = [block(Layerplanes[i+l], Layerplanes[i+1+l], groups = groups)]
                for j in range(num-1):
                    tmp.append(block(Layerplanes[i+1+l], Layerplanes[i+1+l], groups = groups))
                
                if l == 0:
                    self.blocks_1.append(nn.Sequential(*tmp))
                elif l == 1:
                    self.blocks_2.append(nn.Sequential(*tmp))
                elif l == 2:
                    self.blocks_3.append(nn.Sequential(*tmp))
                else:
                    self.blocks_4 = nn.Sequential(*tmp) # last layer only have 1 block

    def forward(self, x):
        x = self.stem(x) 
        x1 = self.downSteps[0](x) # 64 > H/4, W/4

        x1 = self.blocks_1[0](x1)
        x2 = self.downSteps[1](x1)

        x1 = self.blocks_1[1](x1)
        x2 = self.blocks_2[0](x2)
        x3 = self.downSteps[2](x2)
        x1,x2 = switchLayer(self.channels, [x1,x2])

        x1 = self.blocks_1[2](x1)
        x2 = self.blocks_2[1](x2)
        x3 = self.blocks_3[0](x3)
        x4 = self.downSteps[3](x3)
        x1,x2,x3 = switchLayer(self.channels, [x1,x2,x3])

        x1 = self.blocks_1[3](x1)
        x2 = self.blocks_2[2](x2)
        x3 = self.blocks_3[1](x3)
        x4 = self.blocks_4(x4)

        return x1,x2,x3,x4


def FSNet_S(pretrained = True):
    numofblocks = [
        [4,1,1,1],
        [4,2,2],
        [8,8],
        [4]
    ]
    model = FeatureShuffleNet(BasisBlock, channels = 64, numofblocks = numofblocks)
    print("FSNet_S parameter size: ", count_parameters(model))
    if pretrained:
        # load_path = "./pretrained/triHRnet_Synth_weight.pth"
        # cpt = torch.load(load_path)
        # model.load_state_dict(cpt, strict=True)
        # print("load pretrain weight from {}. ".format(load_path))
        print("FSNet_S does not have pretrained weight yet. ")
    return model


def horizonBlock(plane):
    return nn.Sequential(
        nn.Conv2d(plane, plane, (3,9), stride = 1, padding = (1,4)), # (3,15) 7
        nn.ReLU(),
        nn.Conv2d(plane, plane, (3,9), stride = 1, padding = (1,4)),
        nn.ReLU() 
    )


class reduceBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if up:
            self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) 
        else:
            self.deconv = None
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = self.bn2(x)
        x = F.relu(x)
        if self.deconv:
            x = self.deconv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, inplane, kernel_size = 7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(inplane)
        self.sp = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sp(x) * x
        return x


class FPN(nn.Module):

    def __init__(self, backbone='FSNet_M'):
        super().__init__()
        self.backbone_name = backbone
        self.cbam_block = False
        self.hor_block = False

        if backbone in ["FSNet_hor"]:
            self.backbone = FSNet_M(pretrained=False)
            out_channels = self.backbone.channels * 4
            self.hor_block = True
            self.hors = nn.ModuleList()
            for i in range(4):
                self.hors.append(horizonBlock(out_channels))
            self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.reduceLayer = reduceBlock(out_channels * 4, 32, up = True)
            self.skipfpn = True

        elif backbone in ["FSNet_S"]:
            self.backbone = FSNet_S(pretrained=False)
            out_channels = self.backbone.channels * 4
            self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.reduceLayer = reduceBlock(out_channels * 4, 32, up = True)

        elif backbone in ["FSNet_M"]:
            self.backbone = FSNet_M(pretrained=False)
            out_channels = self.backbone.channels * 4
            self.upc1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
            self.reduceLayer = reduceBlock(out_channels * 4, 32, up = True)
            self.cbam_block = True

        else:
            print("backbone is not support !")

        if self.cbam_block:
            self.cbam2 = CBAM(out_channels, kernel_size = 9)
            self.cbam3 = CBAM(out_channels, kernel_size = 7)
            self.cbam4 = CBAM(out_channels, kernel_size = 5)
            self.cbam5 = CBAM(out_channels, kernel_size = 3)

    def upsample(self, x, size):
        _,_,h,w = size
        return F.interpolate(x, size=(h, w), mode='bilinear')

    def forward(self, x):
        c2,c3,c4,c5 = self.backbone(x)
        if self.hor_block:
            c2 = self.hors[0](c2)
            c3 = self.hors[1](c3)
            c4 = self.hors[2](c4)
            c5 = self.hors[3](c5)
        if self.cbam_block:
            c2 = self.cbam2(c2)
            c3 = self.cbam3(c3)
            c4 = self.cbam4(c4)
            c5 = self.cbam5(c5)

        c3 = self.upsample(c3, size=c2.shape)
        c4 = self.upsample(c4, size=c2.shape)
        c5 = self.upsample(c5, size=c2.shape)
        
        c1 = self.upc1(self.reduceLayer(torch.cat([c2,c3,c4,c5], dim=1)))
        del c2
        del c3
        del c4
        del c5
        return c1 


class Positional_encoding(nn.Module):
    def __init__(self, PE_size, n_position=256):
        super(Positional_encoding, self).__init__()
        self.PE_size = PE_size
        self.n_position = n_position
        self.register_buffer('pos_table', self.get_encoding_table(n_position, PE_size))

    def get_encoding_table(self, n_position, PE_size):
        position_table = np.array(
            [[pos / np.power(10000, 2. * i / self.PE_size) for i in range(self.PE_size)] for pos in range(n_position)])
        position_table[:, 0::2] = np.sin(position_table[:, 0::2])
        position_table[:, 1::2] = np.cos(position_table[:, 1::2])
        return torch.FloatTensor(position_table).unsqueeze(0)

    def forward(self, inputs):
        return inputs + self.pos_table[:, :inputs.size(1), :].clone().detach()


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout=0.1, if_resi=True, batch_first=False):
        super(MultiHeadAttention, self).__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.MultiheadAttention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=batch_first)
        self.Q_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.K_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.V_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU())
        self.if_resi = if_resi

    def forward(self, inputs):
        query = self.layer_norm(inputs)
        q = self.Q_proj(query)
        k = self.K_proj(query)
        v = self.V_proj(query)
        attn_output, attn_output_weights = self.MultiheadAttention(q, k, v)
        if self.if_resi:
            attn_output += inputs
        else:
            attn_output = attn_output

        return attn_output


class FeedForward(nn.Module):
    def __init__(self, in_channel, FFN_channel, if_resi=True):
        super(FeedForward, self).__init__()
        """
        1024 2048
        """
        output_channel = (FFN_channel, in_channel)
        self.fc1 = nn.Sequential(nn.Linear(in_channel, output_channel[0]), nn.ReLU())
        self.fc2 = nn.Linear(output_channel[0], output_channel[1])
        self.layer_norm = nn.LayerNorm(in_channel)
        self.if_resi = if_resi

    def forward(self, inputs):
        outputs = self.layer_norm(inputs)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        if self.if_resi:
            outputs += inputs
        else:
            outputs = outputs
        return outputs


class TransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attention_size,
                 dim_feedforward=1024, drop_rate=0.1, if_resi=True, block_nums=3, batch_first = False):
        super(TransformerLayer, self).__init__()
        self.block_nums = block_nums
        self.if_resi = if_resi
        self.linear = nn.Linear(in_dim, attention_size)
        for i in range(self.block_nums):
            self.__setattr__('MHA_self_%d' % i, MultiHeadAttention(num_heads, attention_size, dropout=drop_rate, if_resi=if_resi, batch_first=batch_first))
            self.__setattr__('FFN_%d' % i, FeedForward(out_dim, dim_feedforward, if_resi=if_resi))

    def forward(self, query):
        inputs = self.linear(query)
        # outputs = inputs
        for i in range(self.block_nums):
            outputs = self.__getattr__('MHA_self_%d' % i)(inputs)
            outputs = self.__getattr__('FFN_%d' % i)(outputs)
            if self.if_resi:
                inputs = inputs+outputs
            else:
                inputs = outputs
        # outputs = inputs
        return inputs


class Transformer(nn.Module):

    def __init__(self, in_dim, out_dim, num_heads=8,
                 dim_feedforward=1024, drop_rate=0.1, if_resi=False, block_nums=3,
                 pred_num=2, batch_first=False):
        super().__init__()

        self.bn0 = nn.BatchNorm1d(in_dim, affine=False)
        self.conv1 = nn.Conv1d(in_dim, out_dim, 1, dilation=1)

        self.transformer = TransformerLayer(in_dim, out_dim, num_heads, attention_size=out_dim,
                                            dim_feedforward=dim_feedforward, drop_rate=drop_rate,
                                            if_resi=if_resi, block_nums=block_nums, batch_first=batch_first)
        # self.transformer_contour = TransformerLayer(in_dim, out_dim, num_heads, attention_size=out_dim,
        #                                     dim_feedforward=dim_feedforward, drop_rate=drop_rate,
        #                                     if_resi=if_resi, block_nums=block_nums, batch_first=True)

        self.prediction = nn.Sequential(
            nn.Conv1d(2*out_dim, 128, 1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, pred_num, 1))

    def forward(self, x):
        x = self.bn0(x)

        # ver1
        x1 = x.permute(0, 2, 1)
        x1 = self.transformer(x1)
        x1 = x1.permute(0, 2, 1)
        x = torch.cat([x1, self.conv1(x)], dim=1)
        pred = self.prediction(x)

        # ver2
        # x = x.permute(0, 2, 1)
        # x1 = self.transformer(x)
        # x2 = self.transformer_contour(x)
        # x1 = x1.permute(0, 2, 1)
        # x2 = x2.permute(0, 2, 1)
        # x = torch.cat([x1, x2], dim=1)
        # pred = self.prediction(x)

        return pred


def get_node_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone().float()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
        
    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    # print(img_poly.shape)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        gcn_feature[ind == i] = torch.nn.functional.grid_sample(cnn_feature[i:i + 1], poly).squeeze(0).permute(1, 0, 2)
    return gcn_feature


class midlinePredictor(nn.Module):
    def __init__(self, seg_channel, scale, dis_threshold, cls_threshold, num_points, approx_factor):
        super(midlinePredictor, self).__init__()
        self.seg_channel = seg_channel
        self.scale = scale
        self.dis_threshold = dis_threshold
        self.cls_threshold = cls_threshold
        self.num_points = num_points
        self.approx_factor = approx_factor
        self.clip_dis = 100
        self.midline_preds = nn.ModuleList()
        self.contour_preds = nn.ModuleList()
        self.iter = 3 # 3
        for i in range(self.iter):
            self.midline_preds.append(
                Transformer(
                    seg_channel, 128, num_heads=8, 
                    dim_feedforward=1024, drop_rate=0.0, 
                    if_resi=True, block_nums=3, pred_num=2, batch_first=False)
            )
            self.contour_preds.append(
                Transformer(
                    seg_channel, 128, num_heads=8, 
                    dim_feedforward=1024, drop_rate=0.0, 
                    if_resi=True, block_nums=3, pred_num=2, batch_first=False)
            )
        if not self.training:
            self.iter = 1

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def get_boundary_proposal(self, input=None):
        inds = torch.where(input['ignore_tags'] > 0)
        init_polys = input['proposal_points'][inds]
        return init_polys, inds, None

    def get_boundary_proposal_eval(self, input=None, seg_preds=None):
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, ].detach().cpu().numpy()

        inds = []
        init_polys = []
        confidences = []
        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = dis_pred > self.dis_threshold
            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = round(cls_preds[bid][text_mask].mean(), 3)
                # 50 for MLT2017 and ArT (or DCN is used in backone); else is all 150;
                # just can set to 50, which has little effect on the performance
                if np.sum(text_mask) < 50/(self.scale*self.scale) or confidence < self.cls_threshold:
                    continue
                confidences.append(confidence)
                inds.append([bid, 0])
                
                poly = get_sample_point(text_mask, self.num_points,
                                        self.approx_factor, scales=np.array([self.scale, self.scale]))
                init_polys.append(poly)

        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
        else:
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
            inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)

        return init_polys, inds, confidences

    # def get_boundary_proposal_eval_cuda(self, input=None, seg_preds=None):

    #     # need to return mid line
    #     # print ("using cuda ccl")
    #     cls_preds = seg_preds[:, 0, :, :].detach()
    #     dis_preds = seg_preds[:, 1, :, :].detach()

    #     inds = []
    #     init_polys = []
    #     confidences = []
    #     for bid, dis_pred in enumerate(dis_preds):
    #         dis_mask = dis_pred > cfg.dis_threshold
    #         dis_mask = dis_mask.type(torch.cuda.ByteTensor)
    #         labels = cc_torch.connected_components_labeling(dis_mask)
    #         key = torch.unique(labels, sorted = True)
    #         for l in key:
    #             text_mask = labels == l
    #             confidence = round(torch.mean(cls_preds[bid][text_mask]).item(), 3)
    #             if confidence < cfg.cls_threshold or torch.sum(text_mask) < 10/(cfg.scale*cfg.scale):
    #                 continue
    #             confidences.append(confidence)
    #             inds.append([bid, 0])
                
    #             text_mask = text_mask.cpu().numpy()
    #             poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor, scales=np.array([cfg.scale, cfg.scale]))
    #             init_polys.append(poly)

    #     if len(inds) > 0:
    #         inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
    #     else:
    #         inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)

    #     init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
        
    #     return init_polys, inds, confidences

    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt"):
        if self.training:
            init_polys, inds, confidences = self.get_boundary_proposal(input=input) # get sample point from gt
        else:
            init_polys, inds, confidences = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            # print("iter set to 1 for inference.")
            if init_polys.shape[0] == 0:
                return [init_polys, init_polys], inds, confidences, None
            
        if len(init_polys) == 0:
            py_preds = torch.zeros_like(init_polys)

        h,w = embed_feature.shape[2:4]

        mid_pt_num = init_polys.shape[1] // 2
        contours = [init_polys]
        midlines = []
        for i in range(self.iter):
            node_feat = get_node_feature(embed_feature, contours[i], inds[0], h, w)
            midline = contours[i][:,:mid_pt_num] + torch.clamp(self.midline_preds[i](node_feat).permute(0, 2, 1), -self.clip_dis, self.clip_dis)[:,:mid_pt_num]
            midlines.append(midline)

            mid_feat = get_node_feature(embed_feature, midline, inds[0], h, w)
            node_feat = torch.cat((node_feat, mid_feat), dim=2)
            new_contour = contours[i] + torch.clamp(self.contour_preds[i](node_feat).permute(0, 2, 1), -self.clip_dis, self.clip_dis)[:,:self.num_points]
            contours.append(new_contour)
        
        return contours, inds, confidences, midlines


class Evolution(nn.Module):
    def __init__(self, node_num, seg_channel, scale, dis_threshold, cls_threshold, approx_factor):
        super(Evolution, self).__init__()
        self.num_points = node_num
        self.seg_channel = seg_channel
        self.scale = scale
        self.dis_threshold = dis_threshold
        self.cls_threshold = cls_threshold
        self.approx_factor = approx_factor
        self.clip_dis = 100

        self.iter = 3
        for i in range(self.iter):
            evolve_gcn = Transformer(seg_channel, 128, num_heads=8, dim_feedforward=1024, drop_rate=0.0, if_resi=True, block_nums=3)
            self.__setattr__('evolve_gcn' + str(i), evolve_gcn)
        if not self.training:
            self.iter = 1

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # @staticmethod
    def get_boundary_proposal(self, input=None, seg_preds=None, switch="gt"):

        if switch == "gt":
            inds = torch.where(input['ignore_tags'] > 0)
            init_polys = input['proposal_points'][inds]
        else:
            tr_masks = input['tr_mask'].cpu().numpy()
            tcl_masks = seg_preds[:, 0, :, :].detach().cpu().numpy() > cfg.threshold
            inds = []
            init_polys = []
            for bid, tcl_mask in enumerate(tcl_masks):
                ret, labels = cv2.connectedComponents(tcl_mask.astype(np.uint8), connectivity=8)
                for idx in range(1, ret):
                    text_mask = labels == idx
                    ist_id = int(np.sum(text_mask*tr_masks[bid])/np.sum(text_mask))-1
                    inds.append([bid, ist_id])
                    poly = get_sample_point(text_mask, self.num_points, self.approx_factor)
                    init_polys.append(poly)
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device)

        return init_polys, inds, None

    def get_boundary_proposal_eval(self, input=None, seg_preds=None):
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :, ].detach().cpu().numpy()

        inds = []
        init_polys = []
        confidences = []
        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = dis_pred > self.dis_threshold
            ret, labels = cv2.connectedComponents(dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U)
            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = round(cls_preds[bid][text_mask].mean(), 3)
                # 50 for MLT2017 and ArT (or DCN is used in backone); else is all 150;
                # just can set to 50, which has little effect on the performance
                if np.sum(text_mask) < 50/(self.scale*self.scale) or confidence < self.cls_threshold:
                    continue
                confidences.append(confidence)
                inds.append([bid, 0])
                
                poly = get_sample_point(text_mask, self.num_points,
                                        self.approx_factor, scales=np.array([self.scale, self.scale]))
                init_polys.append(poly)

        if len(inds) > 0:
            inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["image"].device, non_blocking=True)
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["image"].device, non_blocking=True).float()
        else:
            init_polys = torch.from_numpy(np.array(init_polys)).to(input["image"].device, non_blocking=True).float()
            inds = torch.from_numpy(np.array(inds)).to(input["image"].device, non_blocking=True)

        return init_polys, inds, confidences

    # def get_boundary_proposal_eval_cuda(self, input=None, seg_preds=None):
    #     # print ("using cuda ccl")
    #     cls_preds = seg_preds[:, 0, :, :].detach()
    #     dis_preds = seg_preds[:, 1, :, :].detach()

    #     inds = []
    #     init_polys = []
    #     confidences = []
    #     for bid, dis_pred in enumerate(dis_preds):
    #         dis_mask = dis_pred > cfg.dis_threshold
    #         dis_mask = dis_mask.type(torch.cuda.ByteTensor)
    #         labels = cc_torch.connected_components_labeling(dis_mask)
    #         key = torch.unique(labels, sorted = True)
    #         for l in key:
    #             text_mask = labels == l
    #             confidence = round(torch.mean(cls_preds[bid][text_mask]).item(), 3)
    #             if confidence < cfg.cls_threshold or torch.sum(text_mask) < 50/(cfg.scale*cfg.scale):
    #                 continue
    #             confidences.append(confidence)
    #             inds.append([bid, 0])
                
    #             text_mask = text_mask.cpu().numpy()
    #             poly = get_sample_point(text_mask, cfg.num_points, cfg.approx_factor, scales=np.array([cfg.scale, cfg.scale]))
    #             init_polys.append(poly)

    #     if len(inds) > 0:
    #         inds = torch.from_numpy(np.array(inds)).permute(1, 0).to(input["img"].device, non_blocking=True)
    #         init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
    #     else:
    #         init_polys = torch.from_numpy(np.array(init_polys)).to(input["img"].device, non_blocking=True).float()
    #         inds = torch.from_numpy(np.array(inds)).to(input["img"].device, non_blocking=True)

    #     return init_polys, inds, confidences
        
    def evolve_poly(self, snake, cnn_feature, init_poly, ind):
        num_point = init_poly.shape[1]
        if len(init_poly) == 0:
            return torch.zeros_like(init_poly)
        h, w = cnn_feature.size(2)*self.scale, cnn_feature.size(3)*self.scale
        # node_feats: (num_all_text, feat_dim, num_point), sampled from image feature according to coordinates
        node_feats = get_node_feature(cnn_feature, init_poly, ind, h, w)
        i_poly = init_poly + torch.clamp(snake(node_feats).permute(0, 2, 1), -self.clip_dis, self.clip_dis)[:,:num_point]
        if self.training:
            i_poly = torch.clamp(i_poly, 0, w-1)
        else:
            i_poly[:, :, 0] = torch.clamp(i_poly[:, :, 0], 0, w - 1)
            i_poly[:, :, 1] = torch.clamp(i_poly[:, :, 1], 0, h - 1)
        return i_poly

    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt", embed = None):
        # init_polys: (num_all_text, num_point, 2)
        # inds: valid text indexes, (batch index: num_all_text, text in batch index: num_all_text)
        if self.training:
            init_polys, inds, confidences = self.get_boundary_proposal(input=input, seg_preds=seg_preds, switch=switch)
            # TODO sample fix number
        else:
            init_polys, inds, confidences = self.get_boundary_proposal_eval(input=input, seg_preds=seg_preds)
            # init_polys, inds, confidences = self.get_boundary_proposal_eval_cuda(input=input, seg_preds=seg_preds - embed)
            if init_polys.shape[0] == 0:
                return [init_polys for i in range(self.iter+1)], inds, confidences

        py_preds = [init_polys, ]
        for i in range(self.iter):
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            init_polys = self.evolve_poly(evolve_gcn, embed_feature, init_polys, inds[0])
            py_preds.append(init_polys)

        return py_preds, inds, confidences


class TextNet(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config

        self.fpn = FPN(model_config.net)
        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.PReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
        )

        if model_config.embed:
            self.embed_head = nn.Sequential(
                nn.Conv2d(32, 16, kernel_size=3, padding=2, dilation=2),
                nn.PReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
                nn.PReLU(),
                nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
            )
            self.embed_head = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        if not model_config.onlybackbone:
            if model_config.mid:
                self.BPN = midlinePredictor(seg_channel=32+4)
            elif model_config.pos:
                self.BPN = Evolution(
                    model_config.num_points,
                    seg_channel=32+4+2,
                    scale=model_config.scale,
                    dis_threshold=model_config.dis_threshold,
                    cls_threshold=model_config.cls_threshold,
                    approx_factor=model_config.approx_factor,
                )
            else:
                self.BPN = Evolution(
                    model_config.num_points,
                    seg_channel=32+4,
                    scale=model_config.scale,
                    dis_threshold=model_config.dis_threshold,
                    cls_threshold=model_config.cls_threshold,
                    approx_factor=model_config.approx_factor,
                )

    # def load_model(self, model_path):
    #     print('Loading from {}'.format(model_path))
    #     state_dict = torch.load(model_path, map_location=torch.device(cfg.device))
    #     self.load_state_dict(state_dict['model'], strict=(not self.is_training))

    def forward(self, input_dict, test_speed=False, knowledge = False):
        output = {}
        b, c, h, w = input_dict["image"].shape
        
        if self.training:# or cfg.exp_name in ['ArT', 'MLT2017', "MLT2019"] or test_speed:
            image = input_dict["image"]
        else:
            image = input_dict['image'].new_zeros((b, c, self.model_config.test_size[1], self.model_config.test_size[1]), dtype=torch.float32)
            image[:, :, :h, :w] = input_dict["image"][:, :, :, :]

        up1 = self.fpn(image)
        if self.model_config.know or knowledge:
            output["image_feature"] = up1
        if knowledge:
            return output
        preds = self.seg_head(up1)

        fy_preds = torch.cat([torch.sigmoid(preds[:, 0:2, :, :]), preds[:, 2:4, :, :]], dim=1)

        if self.model_config.onlybackbone:
            output["fy_preds"] = fy_preds
            return output

        cnn_feats = torch.cat([up1, fy_preds], dim=1)
        if self.model_config.embed: #or cfg.mid:
            embed_feature = self.embed_head(up1)
            # embed_feature = self.overlap_head(up1)
            # if not self.training:
                # andpart = embed_feature[0][0] * embed_feature[0][1]

        if self.model_config.mid:
            py_preds, inds, confidences, midline = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")
        else:
            py_preds, inds, confidences = self.BPN(cnn_feats, input=input_dict, seg_preds=fy_preds, switch="gt")
        
        output["fy_preds"] = fy_preds
        output["py_preds"] = py_preds
        output["inds"] = inds
        output["confidences"] = confidences
        if self.model_config.mid:
            output["midline"] = midline
        if self.model_config.embed : # or cfg.mid:
            output["embed"] = embed_feature

        return output


class TextLoss(nn.Module):

    def __init__(self, num_points, mid, embed, onlybackbone, scale, max_epoch):
        super().__init__()
        self.num_points = num_points
        self.mid = mid
        self.embed = embed
        self.onlybackbone = onlybackbone
        self.scale = scale
        self.max_epoch = max_epoch
        self.MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)
        self.BCE_loss = torch.nn.BCELoss(reduce=False, size_average=False)
        self.PolyMatchingLoss = PolyMatchingLoss(self.num_points)
        if self.mid:
            self.midPolyMatchingLoss = PolyMatchingLoss(self.num_points // 2)
        # self.embed_loss = EmbLoss_v2()
        # self.ssim = pytorch_ssim.SSIM()
        self.overlap_loss = overlap_loss()

    @staticmethod
    def single_image_loss(pre_loss, loss_label):
        batch_size = pre_loss.shape[0]
        sum_loss = torch.mean(pre_loss.view(-1)) * 0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        eps = 0.001
        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][(loss_label[i] >= eps)])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = torch.mean(pre_loss[i][(loss_label[i] >= eps)])
                sum_loss += posi_loss
                if len(pre_loss[i][(loss_label[i] < eps)]) < 3 * positive_pixel:
                    nega_loss = torch.mean(pre_loss[i][(loss_label[i] < eps)])
                    average_number += len(pre_loss[i][(loss_label[i] < eps)])
                else:
                    nega_loss = torch.mean(torch.topk(pre_loss[i][(loss_label[i] < eps)], 3 * positive_pixel)[0])
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = torch.mean(torch.topk(pre_loss[i], 100)[0])
                average_number += 100
                sum_loss += nega_loss
            # sum_loss += loss/average_number

        return sum_loss/batch_size

    def cls_ohem(self, predict, target, train_mask, negative_ratio=3.):
        pos = (target * train_mask).bool()
        neg = ((1 - target) * train_mask).bool()

        n_pos = pos.float().sum()
        if n_pos.item() > 0:
            loss_pos = self.BCE_loss(predict[pos], target[pos]).sum()
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = min(int(neg.float().sum().item()), int(negative_ratio * n_pos.float()))
        else:
            loss_pos = torch.tensor(0.)
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = 100
        loss_neg, _ = torch.topk(loss_neg, n_neg)

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).float()

    @staticmethod
    def loss_calc_flux(pred_flux, gt_flux, weight_matrix, mask, train_mask):

        # norm loss
        gt_flux = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1).unsqueeze(1) + 1e-3)
        norm_loss = weight_matrix * torch.mean((pred_flux - gt_flux) ** 2, dim=1)*train_mask
        norm_loss = norm_loss.sum(-1).mean()
        # norm_loss = norm_loss.sum()

        # angle loss
        mask = train_mask * mask
        pred_flux = 0.999999 * pred_flux / (pred_flux.norm(p=2, dim=1).unsqueeze(1) + 1e-3)
        # angle_loss = weight_matrix * (torch.acos(torch.sum(pred_flux * gt_flux, dim=1))) ** 2
        # angle_loss = angle_loss.sum(-1).mean()
        angle_loss = (1 - torch.cosine_similarity(pred_flux, gt_flux, dim=1))
        angle_loss = angle_loss[mask].mean()

        return norm_loss, angle_loss

    @staticmethod
    def get_poly_energy(energy_field, img_poly, ind, h, w):
        img_poly = img_poly.clone().float()
        img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
        img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

        batch_size = energy_field.size(0)
        gcn_feature = torch.zeros([img_poly.size(0), energy_field.size(1), img_poly.size(1)]).to(img_poly.device)
        for i in range(batch_size):
            poly = img_poly[ind == i].unsqueeze(0)
            gcn_feature[ind == i] = torch.nn.functional.grid_sample(energy_field[i:i + 1], poly)[0].permute(1, 0, 2)
        return gcn_feature

    def loss_energy_regularization(self, energy_field, img_poly, inds, h, w):
        energys = []
        for i, py in enumerate(img_poly):
            energy = self.get_poly_energy(energy_field.unsqueeze(1), py, inds, h, w)
            energys.append(energy.squeeze(1).sum(-1))

        regular_loss = 0
        energy_loss = 0
        for i, e in enumerate(energys[1:]):
            regular_loss += torch.clamp(e - energys[i], min=0.0).mean()
            energy_loss += torch.where(e <= 0.01, torch.tensor(0.), e).mean()

        return (energy_loss+regular_loss)/len(energys[1:])

    def dice_loss(self, x, target, mask):
        b = x.shape[0]
        x = torch.sigmoid(x)

        x = x.contiguous().reshape(b, -1)
        target = target.contiguous().reshape(b, -1)
        mask = mask.contiguous().reshape(b, -1)

        x = x * mask
        target = target.float()
        target = target * mask

        a = torch.sum(x * target, 1)
        b = torch.sum(x * x, 1) + 0.001
        c = torch.sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)

        loss = 1 - d
        loss = torch.mean(loss)
        return loss

    def forward(self, input_dict, output_dict, eps=None):
        """
          calculate boundary proposal network loss
        """
        # tr_mask = tr_mask.permute(0, 3, 1, 2).contiguous()
        fy_preds = output_dict["fy_preds"]

        if not self.onlybackbone:
            py_preds = output_dict["py_preds"]
            inds = output_dict["inds"]

        train_mask = input_dict['train_mask'].float()    # valid region
        tr_mask = input_dict['tr_mask'] > 0
        distance_field = input_dict['distance_field']
        direction_field = input_dict['direction_field']
        weight_matrix = input_dict['weight_matrix']
        gt_points = input_dict['gt_points']
        instance = input_dict['tr_mask'].long()
        conf = tr_mask.float()
        
        if self.scale > 1:
            train_mask = F.interpolate(train_mask.float().unsqueeze(1),
                                       scale_factor=1/self.scale, mode='bilinear').squeeze().bool()
            tr_mask = F.interpolate(tr_mask.float().unsqueeze(1),
                                    scale_factor=1/self.scale, mode='bilinear').squeeze().bool()

            distance_field = F.interpolate(distance_field.unsqueeze(1),
                                           scale_factor=1/self.scale, mode='bilinear').squeeze()
            direction_field = F.interpolate(direction_field,
                                            scale_factor=1 / self.scale, mode='bilinear')
            weight_matrix = F.interpolate(weight_matrix.unsqueeze(1),
                                          scale_factor=1/self.scale, mode='bilinear').squeeze()

        # pixel class loss
        cls_loss = self.BCE_loss(fy_preds[:, 0, :, :],  conf)
        cls_loss = torch.mul(cls_loss, train_mask).mean()

        # distance field loss
        dis_loss = self.MSE_loss(fy_preds[:, 1, :, :], distance_field)
        dis_loss = torch.mul(dis_loss, train_mask)
        dis_loss = self.single_image_loss(dis_loss, distance_field)

        # # direction field loss
        train_mask = train_mask > 0
        norm_loss, angle_loss = self.loss_calc_flux(fy_preds[:, 2:4, :, :], direction_field, weight_matrix, tr_mask, train_mask)

        if self.onlybackbone:
            alpha = 1.0; beta = 3.0; theta=0.5
            loss = alpha*cls_loss + beta*(dis_loss) + theta*(norm_loss + angle_loss)
            loss_dict = {
                'total_loss': loss,
                'cls_loss': alpha*cls_loss,
                'distance loss': beta*dis_loss,
                'dir_loss': theta*(norm_loss + angle_loss),
                'norm_loss': theta*norm_loss,
            }
            return loss, loss_dict

        # boundary point loss
        point_loss = self.PolyMatchingLoss(py_preds[1:], gt_points[inds])
        if self.mid:
            midline = output_dict["midline"]
            gt_midline = input_dict['gt_mid_points']
            midline_loss = 0.5*self.midPolyMatchingLoss(midline, gt_midline[inds])

        if self.embed:# or cfg.mid:
            embed = output_dict["embed"]
            # kernel_field = distance_field > cfg.dis_threshold
            # embed_loss = self.embed_loss(embed, instance, kernel_field, train_mask.float(), reduce=True)
            # print(input_dict.keys())
            edge_field = input_dict['edge_field'].float()
            # print(edge_field.shape)
            # ssim_loss = 1 - self.ssim(embed, edge_field.unsqueeze(1))
            # embed_loss = self.BCE_loss(embed[:,0,:,:],  edge_field)
            # embed_loss = torch.mul(embed_loss, train_mask).mean() + ssim_loss
            embed_loss = self.overlap_loss(embed, conf, instance, edge_field, inds)
            # if cfg.mid:
            #     embed_loss = embed_loss * 0.1


        #  Minimum energy loss regularization
        h, w = distance_field.size(1) * self.scale, distance_field.size(2) * self.scale
        energy_loss = self.loss_energy_regularization(distance_field, py_preds, inds[0], h, w)

        alpha = 1.0; beta = 3.0; theta=0.5; embed_ratio = 0.5
        if eps is None:
            gama = 0.05; 
        else:
            gama = 0.1 * torch.sigmoid(torch.tensor((eps - self.max_epoch) / self.max_epoch))
        loss = alpha*cls_loss + beta*(dis_loss) + theta*(norm_loss + angle_loss) + gama*(point_loss + energy_loss)
        
        if self.mid:
            loss = loss + gama*midline_loss
        if self.embed: # or cfg.mid:
            loss = loss + embed_ratio*embed_loss

        loss_dict = {
            'total_loss': loss,
            'cls_loss': alpha*cls_loss,
            'distance loss': beta*dis_loss,
            'dir_loss': theta*(norm_loss + angle_loss),
            'norm_loss': theta*norm_loss,
            'angle_loss': theta*angle_loss,
            'point_loss': gama*point_loss,
            'energy_loss': gama*energy_loss,
        }

        if self.embed: # or cfg.mid:
            loss_dict['embed_loss'] = embed_ratio*embed_loss
            # loss_dict['ssim_loss'] = ssim_loss
        if self.mid:
            loss_dict['midline_loss'] = gama*midline_loss

        return loss, loss_dict


class PolyMatchingLoss(nn.Module):
    def __init__(self, pnum, loss_type="L1"):
        super(PolyMatchingLoss, self).__init__()

        self.pnum = pnum
        self.loss_type = loss_type
        self.smooth_L1 = F.smooth_l1_loss
        self.L2_loss = torch.nn.MSELoss(reduce=False, size_average=False)

        batch_size = 1
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=int)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx

        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1)))
        self.register_buffer('feature_id', pidxall.unsqueeze_(2).long().expand(-1, -1, 2))

    def match_loss(self, pred, gt):
        batch_size = pred.shape[0]
        feature_id = self.feature_id.expand(batch_size, -1, -1)

        gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, self.pnum, self.pnum, 2)
        pred_expand = pred.unsqueeze(1)

        if self.loss_type == "L2":
            dis = self.L2_loss(pred_expand, gt_expand)
            dis = dis.sum(3).sqrt().mean(2)
        elif self.loss_type == "L1":
            dis = self.smooth_L1(pred_expand, gt_expand, reduction='none')
            dis = dis.sum(3).mean(2)

        min_dis, min_id = torch.min(dis, dim=1, keepdim=True)

        return min_dis

    def forward(self, pred_list, gt):
        loss = 0
        for pred in pred_list:
            loss += torch.mean(self.match_loss(pred, gt))

        return loss / len(pred_list)


class overlap_loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.MSE_loss = torch.nn.MSELoss(reduce=False, size_average=False)
        self.BCE_loss = torch.nn.BCELoss(reduce=False, size_average=False)
        self.inst_loss = torch.nn.MSELoss(reduction = 'sum')

    def forward(self, preds, conf, inst, overlap, inds):
        p1 = preds[:,0]
        p2 = preds[:,1]
        and_preds = p1 * p2

        and_loss = self.BCE_loss(and_preds, overlap)

        or_preds = torch.maximum(p1,p2)
        or_loss = self.BCE_loss(or_preds, conf)

        and_overlap = and_preds * overlap
        op1 = torch.maximum(p1, and_overlap)
        op2 = torch.maximum(p2, and_overlap)

        inst_loss = torch.tensor(0)
        b, h, w = p1.shape
        for i in range(b):
            bop1 = op1[i]
            bop2 = op2[i]
            inst_label = inst[i]
            keys = torch.unique(inst_label)
            # print(keys.shape)
            tmp = torch.tensor(0)
            for k in keys:
                inst_map = (inst_label == k).float()
                suminst = torch.sum(inst_map)
                d1 = self.inst_loss(bop1 * inst_map, inst_map) / suminst
                d2 = self.inst_loss(bop2 * inst_map, inst_map) / suminst
                tmp = tmp + torch.min(d1,d2) - torch.max(d1,d2) + 1
            inst_loss = inst_loss + ( tmp / keys.shape[0] ) 
        # print(and_loss[conf == 1].mean(), and_loss[conf == 0].mean())
        and_loss = and_loss[conf == 1].mean() + and_loss[conf == 0].mean()
        or_loss = or_loss.mean()
        inst_loss = inst_loss / b
        # print("and_loss : ",and_loss.item(), "or_loss : ",or_loss.item(),"inst_loss : ",inst_loss.item())
        # print("or_loss : ",or_loss.item())
        # print("inst_loss : ",inst_loss.item())
        loss = 0.5 * and_loss +  0.25 * or_loss + 0.25 * inst_loss
        return loss


class PLBase(lightning.LightningModule):
    def __init__(self, model_config, optimizer_config, ckpt_path=None):
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        self.model = TextNet(model_config)
        self.loss_fn = TextLoss(
            model_config.num_points,
            model_config.mid,
            model_config.embed,
            model_config.onlybackbone,
            model_config.scale,
            optimizer_config.max_epochs,
        )
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
    
    def init_from_ckpt(self, path):
        ckpt = torch.load(path, map_location='cpu')
        missing, unexpected = self.load_state_dict(ckpt['state_dict'], strict=False)
        print(f'Restored from checkpoint {ckpt}')
        print(f'Missing keys: {missing}')
        print(f'Unexpected keys: {unexpected}')
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        loss, logdict = self.loss_fn(batch, outputs, eps=self.current_epoch + 1)
        logdict = {f'train/{k}': v for k, v in logdict.items()}
        self.log_dict(logdict, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=batch['image'].size(0))
        
        os.makedirs(self.logger.log_dir, exist_ok=True)

        if batch_idx % 100 == 0:
            fy_preds = F.interpolate(outputs["fy_preds"], scale_factor=self.model_config.scale, mode='bilinear')
            fy_preds = fy_preds.cpu().numpy()

            py_preds = outputs["py_preds"][1:]
            init_polys = outputs["py_preds"][0]
            inds = outputs["inds"]

            if self.model_config.mid == True:
                midline = outputs["midline"]

            image = batch['img']
            tr_mask = batch['tr_mask'].cpu().numpy() > 0
            distance_field = batch['distance_field'].data.cpu().numpy()
            direction_field = batch['direction_field']
            weight_matrix = batch['weight_matrix']
            gt_tags = batch['gt_points'].cpu().numpy()
            ignore_tags = batch['ignore_tags'].cpu().numpy()

            b, c, _, _ = fy_preds.shape
            for i in range(b):

                fig = plt.figure(figsize=(12, 9))

                mask_pred = fy_preds[i, 0, :, :]
                distance_pred = fy_preds[i, 1, :, :]
                norm_pred = np.sqrt(fy_preds[i, 2, :, :] ** 2 + fy_preds[i, 3, :, :] ** 2)
                angle_pred = 180 / math.pi * np.arctan2(fy_preds[i, 2, :, :], fy_preds[i, 3, :, :] + 0.00001)

                ax1 = fig.add_subplot(341)
                ax1.set_title('mask_pred')
                im1 = ax1.imshow(mask_pred, cmap=cm.jet)

                ax2 = fig.add_subplot(342)
                ax2.set_title('distance_pred')
                im2 = ax2.imshow(distance_pred, cmap=cm.jet)

                ax3 = fig.add_subplot(343)
                ax3.set_title('norm_pred')
                im3 = ax3.imshow(norm_pred, cmap=cm.jet)

                ax4 = fig.add_subplot(344)
                ax4.set_title('angle_pred')
                im4 = ax4.imshow(angle_pred, cmap=cm.jet)

                mask_gt = tr_mask[i]
                distance_gt = distance_field[i]
                # gt_flux = 0.999999 * direction_field[i] / (direction_field[i].norm(p=2, dim=0) + 1e-9)
                gt_flux = direction_field[i].cpu().numpy()
                norm_gt = np.sqrt(gt_flux[0, :, :] ** 2 + gt_flux[1, :, :] ** 2)
                angle_gt = 180 / math.pi * np.arctan2(gt_flux[0, :, :], gt_flux[1, :, :]+0.00001)

                ax11 = fig.add_subplot(345)
                im11 = ax11.imshow(mask_gt, cmap=cm.jet)

                ax22 = fig.add_subplot(346)
                im22 = ax22.imshow(distance_gt, cmap=cm.jet)

                ax33 = fig.add_subplot(347)
                im33 = ax33.imshow(norm_gt, cmap=cm.jet)

                ax44 = fig.add_subplot(348)
                im44 = ax44.imshow(angle_gt, cmap=cm.jet)

                img_show = image[i].permute(1, 2, 0).cpu().numpy()
                img_show = ((img_show * self.model_config.stds + self.model_config.means) * 255).astype(np.uint8)
                img_show = np.ascontiguousarray(img_show[:, :, ::-1])
                shows = []
                gt = gt_tags[i]
                gt_idx = np.where(ignore_tags[i] > 0)
                gt_py = gt[gt_idx[0], :, :]
                index = torch.where(inds[0] == i)[0]
                init_py = init_polys[index].detach().cpu().numpy()

                image_show = img_show.copy()
                cv2.drawContours(image_show, init_py.astype(np.int32), -1, (255, 0, 0), 2)
                cv2.drawContours(image_show, gt_py.astype(np.int32), -1, (0, 255, 255), 2)
                shows.append(image_show)
                for py in py_preds:
                    contours = py[index].detach().cpu().numpy()
                    image_show = img_show.copy()
                    cv2.drawContours(image_show, init_py.astype(np.int32), -1, (0, 125, 125), 2)
                    cv2.drawContours(image_show, gt_py.astype(np.int32), -1, (255, 125, 0), 2)
                    cv2.drawContours(image_show, contours.astype(np.int32), -1, (0, 255, 125), 2)
                    if self.model_config.mid == True:
                        cv2.polylines(image_show, midline.astype(np.int32), False, (125, 255, 0), 2)
                    shows.append(image_show)

                for idx, im_show in enumerate(shows):
                    axb = fig.add_subplot(3, 4, 9+idx)
                    # axb.set_title('boundary_{}'.format(idx))
                    # axb.set_autoscale_on(True)
                    im11 = axb.imshow(im_show, cmap=cm.jet)
                    # plt.colorbar(im11, shrink=0.5)

                plt.savefig(os.path.join(self.logger.log_dir, f'{i}.png'))
                plt.close(fig)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        outputs.update(batch)
        img_show, contours = self.log_images(outputs, batch_idx)
        outputs['contours'] = contours
        self.log_results(outputs)
    
    def configure_optimizers(self):
        if self.optimizer_config.optim == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.optimizer_config.learning_rate
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.optimizer_config.learning_rate,
                momentum=self.optimizer_config.momentum
            )
        if self.optimizer_config.lr_adjust == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'epoch'
                }
            }
        else:
            return optimizer

    @torch.no_grad()
    def log_images(self, batch, batch_idx):
        dirname = os.path.join(self.logger.log_dir, 'log_images', mode)
        img_show = batch['image'][0].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * self.model_config.stds + self.model_config.means) * 255).astype(np.uint8)
        
        gt_contour = []
        label_tag = batch['label_tag'][0].int().cpu().numpy()
        for anno, n_anno in zip(batch['annotation'][0], batch['n_annotation'][0]):
            if n_anno.item() > 0:
                gt_contour.append(anno[:n_anno].int().cpu().numpy())

        gt_vis = self.visualize_gt(img_show, gt_contour, label_tag)
        show_boundary, heatmap = self.visualize_detection(img_show, batch)
        show_map = np.concatenate([heatmap, gt_vis], axis=1)
        show_map = cv2.resize(show_map, (320 * 3, 320))
        im_vis = np.concatenate([show_map, show_boundary], axis=0)
        cv2.imwrite(os.path.join(dirname, batch['image_id'][0].split('.')[0] + '.jpg'), im_vis)

        contours = batch['py_preds'][-1].int().cpu().numpy()
        H, W = batch['Height'][0].item(), batch['Width'][0].item()
        
        def rescale_result(image, bbox_contours, H, W):
            ori_H, ori_W = image.shape[:2]
            image = cv2.resize(image, (W, H))
            contours = list()
            for cont in bbox_contours:
                # if cv2.contourArea(cont) < 300:
                #     continue
                cont[:, 0] = (cont[:, 0] * W / ori_W).astype(int)
                cont[:, 1] = (cont[:, 1] * H / ori_H).astype(int)
                contours.append(cont)
            return image, contours
        
        img_show, contours = rescale_result(img_show, contours, H, W)

        return img_show, contours
    
    def visualize_gt(self, image, contours, label_tag):
        image_show = image.copy()
        image_show = np.ascontiguousarray(image_show[:, :, ::-1])
        image_show = cv2.polylines(image_show,
                                [contours[i] for i, tag in enumerate(label_tag) if tag >0], True, (0, 0, 255), 3)
        image_show = cv2.polylines(image_show,
                                [contours[i] for i, tag in enumerate(label_tag) if tag <0], True, (0, 255, 0), 3)
        show_gt = cv2.resize(image_show, (320, 320))
        return show_gt
    
    def visualize_detection(self, image, output_dict):
        image_show = image.copy()
        image_show = np.ascontiguousarray(image_show[:, :, ::-1])

        cls_preds = F.interpolate(output_dict["fy_preds"], scale_factor=self.model_cfg.scale, mode='bilinear')
        cls_preds = cls_preds[0].detach().cpu().numpy()
        py_preds = output_dict["py_preds"][1:]
        init_polys = output_dict["py_preds"][0]
        shows = []
        if self.model_cfg.mid:
            midline = output_dict["midline"]

        init_py = init_polys.detach().cpu().numpy()
        path = os.path.join(self.logger.log_dir, 'vis', output_dict['image_id'][0].split(".")[0] + "_init.png")
        im_show0 = image_show.copy()
        for i, bpts in enumerate(init_py.astype(np.int32)):
            cv2.drawContours(im_show0, [bpts.astype(np.int32)], -1, (255, 0, 255), 2)
            for j, pp in enumerate(bpts):
                if j == 0:
                    cv2.circle(im_show0, (int(pp[0]), int(pp[1])), 3, (125, 125, 255), -1)
                elif j == 1:
                    cv2.circle(im_show0, (int(pp[0]), int(pp[1])), 3, (125, 255, 125), -1)
                else:
                    cv2.circle(im_show0, (int(pp[0]), int(pp[1])), 3, (255, 125, 125), -1)
        cv2.imwrite(path, im_show0)

        for idx, py in enumerate(py_preds):
            im_show = im_show0.copy()
            contours = py.detach().cpu().numpy()
            cv2.drawContours(im_show, contours.astype(np.int32), -1, (0, 255, 255), 2)
            for ppts in contours:
                for j, pp in enumerate(ppts):
                    if j == 0:
                        cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (125, 125, 255), -1)
                    elif j == 1:
                        cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (125, 255, 125), -1)
                    else:
                        cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (255, 125, 125), -1)
            if self.model_config.mid:
                for ppt in midline:
                    for pt in ppt:
                        cv2.circle(im_show, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

            path = os.path.join(self.logger.log_dir, 'vis', output_dict['image_id'][0].split(".")[0] + "_{}iter.png".format(idx))
            cv2.imwrite(path, im_show)
            shows.append(im_show)

        show_img = np.concatenate(shows, axis=1)
        show_boundary = cv2.resize(show_img, (320 * len(py_preds), 320))

        def heatmap(im_gray):
            cmap = plt.get_cmap('jet')
            rgba_img = cmap(255 - im_gray)
            Hmap = np.delete(rgba_img, 3, 2)
            # print(Hmap.shape, Hmap.max(), Hmap.min())
            # cv2.imshow("heat_img", Hmap)
            # cv2.waitKey(0)
            return Hmap

        cls_pred = heatmap(np.array(cls_preds[0] * 255, dtype=np.uint8))
        dis_pred = heatmap(np.array(cls_preds[1] * 255, dtype=np.uint8))

        heat_map = np.concatenate([cls_pred*255, dis_pred*255], axis=1)
        heat_map = cv2.resize(heat_map, (320 * 2, 320))

        return show_boundary, heat_map

    @torch.no_grad()
    def log_results(self, batch):
        contours = np.array(contours).astype(int)
        contours = np.expand_dims(contours, axis=2)
        with open(os.path.join(self.logger.log_dir, 'results', batch['image_id'][0].replace('jpg', 'txt'))) as f:
            for contour in contours:
                contour = np.stack([contour[:, 0], contour[:, 1]], 1)
                if cv2.contourArea(contour) <= 0:
                    continue
                contour = contour.flatten().astype(str).tolist()
                contour = ','.join(contour)
                f.write(contour + '\n')