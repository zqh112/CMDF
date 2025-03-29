import math
from collections import deque
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
from torch.nn.parameter import Parameter
class ImageCNN(nn.Module):
    """
    Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
    """
    def __init__(self, c_dim, normalize=True):
        super().__init__()
        self.normalize = normalize
        self.features = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.features.fc = nn.Sequential()
        # for param in self.features.parameters():
        #     param.requires_grad = False

    def forward(self, inputs):
        c = 0
        for x in inputs:
            if self.normalize:
                x = normalize_imagenet(x)
            c += self.features(x)
        return c

def normalize_imagenet(x):
    """ Normalize input images according to ImageNet standards.
    Args:
        x (tensor): input images
    """
    x = x.clone()
    x[:, 0] = (x[:, 0]/255.0 - 0.485) / 0.229
    x[:, 1] = (x[:, 1]/255.0 - 0.456) / 0.224
    x[:, 2] = (x[:, 2]/255.0 - 0.406) / 0.225
    return x


class RadarEncoder(nn.Module):
    """
    Encoder network for LiDAR input list
    Args:
        num_classes: output feature dimension
        in_channels: input channels
    """

    def __init__(self, num_classes=512, in_channels=2):
        super().__init__()

        self._model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self._model.fc = nn.Sequential()
        _tmp = self._model.conv1
        self._model.conv1 = nn.Conv2d(in_channels, out_channels=_tmp.out_channels,
            kernel_size=_tmp.kernel_size, stride=_tmp.stride, padding=_tmp.padding, bias=_tmp.bias)
        # for param in self._model.parameters():
        #     param.requires_grad = False

    def forward(self, inputs):
        features = 0
        for radar_data in inputs:
            radar_feature = self._model(radar_data)
            features += radar_feature
        return features

class ECA_layer(nn.Module):
    def __init__(self, x, gamma=2, bias=1):
        super(ECA_layer, self).__init__()
        self.x = x
        self.gamma = gamma
        self.bias = bias
        n, c, h, w= x.size()
        t = int(abs((math.log(c, 2) + self.bias) / self.gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self):
        y = self.avg_pool(self.x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return self.x * y.expand_as(self.x)

class CrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.img_key = nn.Linear(n_embd, n_embd)
        self.img_query = nn.Linear(n_embd, n_embd)
        self.img_value = nn.Linear(n_embd, n_embd)

        self.radar_key = nn.Linear(n_embd, n_embd)
        self.radar_query = nn.Linear(n_embd, n_embd)
        self.radar_value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        x_img = x[0]
        x_radar = x[1]
        B, T, C = x_img.shape[:3]

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k_img = self.img_key(x_img).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q_img = self.img_query(x_img).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v_img = self.img_value(x_img).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        k_radar = self.radar_key(x_radar).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q_radar = self.radar_query(x_radar).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v_radar = self.radar_value(x_radar).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att_img = (q_radar @ k_img.transpose(-2, -1)) * (1.0 / math.sqrt(k_img.size(-1)))
        att_radar = (q_img @ k_radar.transpose(-2, -1)) * (1.0 / math.sqrt(k_radar.size(-1)))

        att_img = F.softmax(att_img, dim=-1)
        att_img = self.attn_drop(att_img)
        att_radar = F.softmax(att_radar, dim=-1)
        att_radar = self.attn_drop(att_radar)

        out_img = att_img @ v_img  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out_img = out_img.transpose(1, 2).contiguous().view(B, T, C)
        out_radar = att_radar @ v_radar
        out_radar = out_radar.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        out_img = self.resid_drop(self.proj(out_img))
        out_radar = self.resid_drop(self.proj(out_radar))
        return [out_img, out_radar]


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CrossAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x_img = x[0]
        x_radar = x[1]
        x_img_ln = self.ln1(x_img)
        x_radar_ln = self.ln1(x_radar)
        loops_num = 1
        for loop in range(loops_num):
            out = self.attn([x_img_ln, x_radar_ln])
            x_img_out = x_img + out[0]
            x_radar_out = x_radar + out[1]
            x_img_ln = x_img_out + self.mlp(self.ln2(x_img_out))
            x_radar_ln = x_radar_out + self.mlp(self.ln2(x_radar_out))
        return [x_img_ln, x_radar_ln]


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 vert_anchors, horz_anchors, seq_len,
                 embd_pdrop, attn_pdrop, resid_pdrop, config):
        super().__init__()
        self.n_embd = n_embd
        self.seq_len = seq_len
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image
        self.pos_emb = nn.Parameter(
            torch.zeros(1, self.config.n_views * seq_len * vert_anchors * horz_anchors, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop)
                                      for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = seq_len
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # create the pytorch optimizer object
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        return optim_groups

    def forward(self, image_tensor, radar_tensor):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            gps (tensor): ego-gps
        """

        bz = image_tensor.shape[0] // self.seq_len
        h, w = image_tensor.shape[2:4]
        #         print('transfo',self.config.n_views , self.seq_len)

        # forward the image model for token embeddings
        image_tensor = image_tensor.view(bz, self.config.n_views * self.seq_len, -1, h, w)
        radar_tensor = radar_tensor.view(bz, self.seq_len, -1, h, w)

        # pad token embeddings along number of tokens dimension
        image_embeddings = image_tensor.permute(0, 1, 3, 4, 2).contiguous()
        image_embeddings = image_embeddings.view(bz, -1, self.n_embd)  # (B, an * T, C)
        radar_embeddings = radar_tensor.permute(0, 1, 3, 4, 2).contiguous()
        radar_embeddings = radar_embeddings.view(bz, -1, self.n_embd)  # (B, an * T, C)

        # add (learnable) positional embedding and gps embedding for all tokens
        x_image = self.drop(self.pos_emb + image_embeddings)  # (B, an * T, C)
        x_radar = self.drop(self.pos_emb + radar_embeddings)  # (B, an * T, C)
        out = self.blocks([x_image, x_radar])
        out_image, out_radar = out[0], out[1]  # (B, an * T, C)
        out_image = self.ln_f(out_image).contiguous().view(bz * self.config.n_views * self.seq_len, -1, h, w)  # (B, an * T, C)
        out_radar = self.ln_f(out_radar).contiguous().view(bz * self.seq_len, -1, h, w)  # (B, an * T, C)

        return out_image, out_radar


class Encoder(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device

        self.avgpool = nn.AdaptiveAvgPool2d((self.config.vert_anchors, self.config.horz_anchors))
        self.image_encoder = ImageCNN(512, normalize=True)
        if config.add_velocity:
            self.radar_encoder = RadarEncoder(num_classes=512, in_channels=2)
        else:
            self.radar_encoder = RadarEncoder(num_classes=512, in_channels=1)

        self.transformer2 = GPT(n_embd=128,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer3 = GPT(n_embd=256,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)
        self.transformer4 = GPT(n_embd=512,
                                n_head=config.n_head,
                                block_exp=config.block_exp,
                                n_layer=config.n_layer,
                                vert_anchors=config.vert_anchors,
                                horz_anchors=config.horz_anchors,
                                seq_len=config.seq_len,
                                embd_pdrop=config.embd_pdrop,
                                attn_pdrop=config.attn_pdrop,
                                resid_pdrop=config.resid_pdrop,
                                config=config)

    def forward(self, image_list, radar_list):
        '''
        Image + LiDAR feature fusion using transformers
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            gps (tensor): input gps
        '''
        if self.image_encoder.normalize:
            image_list = [normalize_imagenet(image_input) for image_input in image_list]

        bz, _, h, w = image_list[0].shape
        img_channel = image_list[0].shape[1]
        radar_channel = radar_list[0].shape[1]

        self.config.n_views = len(image_list) // self.config.seq_len

        image_tensor = torch.stack(image_list, dim=1).view(bz * self.config.n_views * self.config.seq_len, img_channel, h, w)
        radar_tensor = torch.stack(radar_list, dim=1).view(bz * self.config.seq_len, radar_channel, h, w)

        image_features = self.image_encoder.features.conv1(image_tensor)
        image_features = self.image_encoder.features.bn1(image_features)
        image_features = self.image_encoder.features.relu(image_features)
        image_features = self.image_encoder.features.maxpool(image_features)

        radar_features = self.radar_encoder._model.conv1(radar_tensor)
        radar_features = self.radar_encoder._model.bn1(radar_features)
        radar_features = self.radar_encoder._model.relu(radar_features)
        radar_features = self.radar_encoder._model.maxpool(radar_features)

        image_features = self.image_encoder.features.layer1(image_features)
        radar_features = self.radar_encoder._model.layer1(radar_features)

        image_features = self.image_encoder.features.layer2(image_features)
        radar_features = self.radar_encoder._model.layer2(radar_features)
        image_embd_layer2 = self.avgpool(image_features)
        radar_embd_layer2 = self.avgpool(radar_features)

        image_features_layer2, radar_features_layer2 = self.transformer2(image_embd_layer2, radar_embd_layer2)
        image_features_layer2 = F.interpolate(image_features_layer2, scale_factor=4, mode='bilinear')
        radar_features_layer2 = F.interpolate(radar_features_layer2, scale_factor=4, mode='bilinear')
        image_features = image_features + image_features_layer2
        radar_features = radar_features + radar_features_layer2

        image_features = self.image_encoder.features.layer3(image_features)
        radar_features = self.radar_encoder._model.layer3(radar_features)
        image_embd_layer3 = self.avgpool(image_features)
        radar_embd_layer3 = self.avgpool(radar_features)

        image_features_layer3, radar_features_layer3 = self.transformer3(image_embd_layer3, radar_embd_layer3)
        image_features_layer3 = F.interpolate(image_features_layer3, scale_factor=2, mode='bilinear')
        radar_features_layer3 = F.interpolate(radar_features_layer3, scale_factor=2, mode='bilinear')
        image_features = image_features + image_features_layer3
        radar_features = radar_features + radar_features_layer3

        image_features = self.image_encoder.features.layer4(image_features)
        radar_features = self.radar_encoder._model.layer4(radar_features)
        image_embd_layer4 = self.avgpool(image_features)
        radar_embd_layer4 = self.avgpool(radar_features)

        image_features_layer4, radar_features_layer4 = self.transformer4(image_embd_layer4, radar_embd_layer4)
        image_features = image_features + image_features_layer4
        radar_features = radar_features + radar_features_layer4

        self.eca_fusion1 = ECA_layer(image_features, gamma=2, bias=1).to(self.device)
        self.eca_fusion2 = ECA_layer(radar_features, gamma=2, bias=1).to(self.device)

        image_features = self.eca_fusion1()
        radar_features = self.eca_fusion2()

        image_features = self.image_encoder.features.avgpool(image_features)
        image_features = torch.flatten(image_features, 1)
        image_features = image_features.view(bz, self.config.n_views * self.config.seq_len, -1)
        image_features = torch.sum(image_features, dim=1)

        radar_features = self.radar_encoder._model.avgpool(radar_features)
        radar_features = torch.flatten(radar_features, 1)
        radar_features = radar_features.view(bz, self.config.seq_len, -1)
        radar_features = torch.sum(radar_features, dim=1)

        return image_features, radar_features


class CMDF(nn.Module):
    '''
    Transformer-based feature fusion followed by GRU-based waypoint prediction network and PID controller
    '''

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.pred_len = config.pred_len
        self.encoder = Encoder(config, device).to(self.device)

        self.join = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
        ).to(self.device)

    def forward(self, image_list, radar_list):
        '''
        Predicts waypoint from geometric feature projections of image + LiDAR input
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
            target_point (tensor): goal location registered to ego-frame
            gps (tensor): input gps
        '''
        image, radar = self.encoder(image_list, radar_list)
        image_out = self.join(image)
        radar_out = self.join(radar)

        image_energy = -torch.logsumexp(image_out, dim=1)
        radar_energy = -torch.logsumexp(radar_out, dim=1)

        image_conf = -0.1 * torch.reshape(image_energy, (-1, 1))
        radar_conf = -0.1 * torch.reshape(radar_energy, (-1, 1))
        fused_out = image_out * image_conf + radar_out * radar_conf

        return fused_out, image_out, radar_out, image_conf, radar_conf
