import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from .blocks import FeatureFusionBlock_custom, Interpolate, _make_encoder, _make_scratch, _make_pretrained_efficientnet_lite3
import os


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)



        self.apply(init_func)

class HazeRemovalNet(BaseNetwork):
    def __init__(self, base_channel_nums, in_channels=3, out_channels=3, init_weights=True, norm_type='batch', min_beta=0.04, max_beta=0.2):
        super(HazeRemovalNet, self).__init__()

        act_type = 'leakyrelu'
        norm_type = norm_type
        mode = 'CNA'
        use_spectral_norm = False

        self.MIN_BETA=min_beta
        self.MAX_BETA=max_beta


        self.enc_conv0 = conv_block(in_nc=in_channels, out_nc=base_channel_nums // 2, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=None,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv1 = conv_block(in_nc=base_channel_nums // 2, out_nc=base_channel_nums, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv2 = conv_block(in_nc=base_channel_nums, out_nc=2 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)

        self.enc_conv3 = conv_block(in_nc=2 * base_channel_nums, out_nc=4 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type,
                                    use_spectral_norm=use_spectral_norm)
        #
        self.bottleneck1 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3, pad_type='reflect',
                                       act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck2 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck3 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)
        self.bottleneck4 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode,
                                       use_spectral_norm=use_spectral_norm)

        self.dec_up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv1 = conv_block(base_channel_nums * 8, out_nc=base_channel_nums * 2, kernel_size=3,
                                    act_type=act_type, norm_type=norm_type, pad_type='reflect', mode=mode,
                                    use_spectral_norm=use_spectral_norm)

        self.dec_up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = conv_block(base_channel_nums * 4, out_nc=base_channel_nums * 1, kernel_size=3,
                                    act_type=act_type, norm_type=None, pad_type='reflect', mode=mode,
                                    use_spectral_norm=use_spectral_norm)

        self.dec_conv_last = conv_block(base_channel_nums * 2, out_nc=base_channel_nums, kernel_size=3,
                                        pad_type='reflect',
                                        act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.dec_conv_last_2 = conv_block(base_channel_nums * 1, out_nc=out_channels, kernel_size=3,
                                          pad_type='reflect',
                                          act_type=None, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.final_conv_beta_1 = conv_block(in_nc= int(base_channel_nums * (1+2+4)), out_nc=2 * base_channel_nums, kernel_size=1,
                                            act_type=None, norm_type=None, use_spectral_norm=use_spectral_norm)
        self.final_conv_beta_2 = conv_block(in_nc=2 * base_channel_nums, out_nc=1, kernel_size=1, act_type=None,
                                            norm_type=None, use_spectral_norm=use_spectral_norm)

        if init_weights:
            self.init_weights('xaiver')

    def forward(self, x , require_paras=False):
        x0 =x
        x = self.enc_conv0(x)
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(x1)
        x3 = self.enc_conv3(x2)

        x = self.bottleneck1(x3)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)

        x = self.dec_up1(torch.cat([x, x3], dim=1))
        x = self.dec_conv1(x)
        x = self.dec_up2(torch.cat([x, x2], dim=1))
        x = self.dec_conv2(x)
        x = self.dec_conv_last(torch.cat([x, x1], dim=1))
        x = self.dec_conv_last_2(x)
        x = torch.tanh(x)
        x = (x0+x).clamp(0,1)

        layer_1_beta = F.adaptive_avg_pool2d(x1, [1, 1]).detach()
        layer_2_beta = F.adaptive_avg_pool2d(x2, [1, 1]).detach()
        layer_3_beta = F.adaptive_avg_pool2d(x3, [1, 1]).detach()
        beta = self.final_conv_beta_1(torch.cat([layer_1_beta, layer_2_beta, layer_3_beta], dim=1))
        beta = self.final_conv_beta_2(beta)
        beta = self.MIN_BETA + (self.MAX_BETA-self.MIN_BETA)*(torch.tanh(beta) + 1) / 2

        if require_paras:
            return x, beta
        else:
            return x

class HazeProduceNet(BaseNetwork):
    def __init__(self, base_channel_nums, in_channels=3, out_channels=3, init_weights=True, norm_type='batch', min_beta=0.04, max_beta=0.2):
        super(HazeProduceNet, self).__init__()

        act_type = 'leakyrelu'
        norm_type = norm_type
        mode = 'CNA'
        use_spectral_norm = False
        self.MAX_BETA = max_beta
        self.MIN_BETA = min_beta
        self.transmission_estimator = TransmissionEstimator()

        self.enc_conv0 = conv_block(in_nc=in_channels, out_nc=base_channel_nums // 2, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.enc_conv1 = conv_block(in_nc=base_channel_nums//2, out_nc=base_channel_nums, kernel_size=3, stride=1,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type, use_spectral_norm=use_spectral_norm)

        self.enc_conv2 = conv_block(in_nc=base_channel_nums, out_nc=2 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type, use_spectral_norm=use_spectral_norm)

        self.enc_conv3 = conv_block(in_nc=2 * base_channel_nums, out_nc=4 * base_channel_nums, kernel_size=3, stride=2,
                                    pad_type='reflect', mode=mode, act_type=act_type, norm_type=norm_type, use_spectral_norm=use_spectral_norm)
        #
        self.bottleneck1 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3, pad_type='reflect',
                                       act_type=act_type, norm_type=norm_type, mode=mode, use_spectral_norm=use_spectral_norm)
        self.bottleneck2 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode, use_spectral_norm=use_spectral_norm)
        self.bottleneck3 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode, use_spectral_norm=use_spectral_norm)
        self.bottleneck4 = ResNetBlock(in_nc=4 * base_channel_nums, mid_nc=4 * base_channel_nums,
                                       out_nc=4 * base_channel_nums, kernel_size=3,
                                       pad_type='reflect', act_type=act_type, norm_type=norm_type, mode=mode, use_spectral_norm=use_spectral_norm)

        self.dec_up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv1 = conv_block(base_channel_nums * 8, out_nc=base_channel_nums * 2, kernel_size=3, act_type=act_type, norm_type=norm_type, pad_type='reflect', mode=mode, use_spectral_norm=use_spectral_norm)

        self.dec_up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = conv_block(base_channel_nums * 4, out_nc=base_channel_nums * 1, kernel_size=3, act_type=act_type, norm_type=None, pad_type='reflect', mode=mode, use_spectral_norm=use_spectral_norm)

        self.dec_conv_last = conv_block(base_channel_nums * 2, out_nc=base_channel_nums, kernel_size=3, pad_type='reflect',
                                        act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        self.dec_conv_last_2 = conv_block(base_channel_nums * 1, out_nc=out_channels, kernel_size=3,
                                          pad_type='reflect',
                                          act_type=act_type, norm_type=None, use_spectral_norm=use_spectral_norm)

        if init_weights:
            self.init_weights('xaiver')

    def forward(self, x, d, beta, A=None, requires_direct_fog=False):
        if A is None:
            A = self.transmission_estimator.get_atmosphere_light_new(x)
        t = torch.exp(-d*beta)
        t = t.clamp(0.1,0.95)

        
        x_clean = t*x+A*(1-t).clamp(0,1)

        x = self.enc_conv0(x_clean)
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(x1)
        x3 = self.enc_conv3(x2)

        x = self.bottleneck1(x3)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)

        x = self.dec_up1(torch.cat([x, x3], dim=1))
        x = self.dec_conv1(x)
        x = self.dec_up2(torch.cat([x, x2], dim=1))
        x = self.dec_conv2(x)
        x = self.dec_conv_last(torch.cat([x, x1], dim=1))
        x = self.dec_conv_last_2(x)
        x = torch.tanh(x)
        x = (x_clean + x).clamp(0, 1)
        if requires_direct_fog:
            return x, x_clean
        else:
            return x

    def forward_random_parameters(self, x, d, requires_direct_fog=False): #x:NCHW, ex:e^(-d(x)), N,1,H,W, beta:N,1,1,1
        n,c,h,w = x.shape
        beta = self.MIN_BETA + torch.rand(n, 1, 1, 1).cuda() * (self.MAX_BETA - self.MIN_BETA)

        if requires_direct_fog:
            res, res_redirect = self(x, d, beta, requires_direct_fog=requires_direct_fog)
            return res, res_redirect, beta
        else:
            res = self(x, d, beta, requires_direct_fog=requires_direct_fog)
            return res, beta


    def direct_forward(self, x):
        x_clean = x
        x = self.enc_conv0(torch.cat([x_clean, x_clean], dim=1))
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(x1)
        x3 = self.enc_conv3(x2)
        x = self.bottleneck1(x3)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)

        x = self.dec_up1(torch.cat([x, x3], dim=1))
        x = self.dec_conv1(x)
        x = self.dec_up2(torch.cat([x, x2], dim=1))
        x = self.dec_conv2(x)
        x = self.dec_conv_last(torch.cat([x, x1], dim=1))

        x = self.dec_conv_last_2(x)
        x = (x_clean + x).clamp(0, 1)

        return x



def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA', use_spectral_norm=False):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC', 'CAN'], 'Wrong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = spectral_norm(nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups), mode=use_spectral_norm)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

    elif mode =='CAN':
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, a, n)




def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ResNetBlock(nn.Module):
    '''
    ResNet Block, 3-3 style
    with extra residual scaling used in EDSR
    (Enhanced Deep Residual Networks for Single Image Super-Resolution, CVPRW 17)
    '''

    def __init__(self, in_nc, mid_nc, out_nc, kernel_size=3, stride=1, dilation=1, groups=1, \
            bias=True, pad_type='zero', norm_type=None, act_type='relu', mode='CNA', res_scale=1, use_spectral_norm=False):
        super(ResNetBlock, self).__init__()
        conv0 = conv_block(in_nc, mid_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode,use_spectral_norm=use_spectral_norm)
        if mode == 'CNA' or mode == 'CAN':
            act_type = None
        if mode == 'CNAC':  # Residual path: |-CNAC-|
            act_type = None
            norm_type = None
        conv1 = conv_block(mid_nc, out_nc, kernel_size, stride, dilation, groups, bias, pad_type, \
            norm_type, act_type, mode, use_spectral_norm=use_spectral_norm)
        self.res = sequential(conv0, conv1)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.res(x).mul(self.res_scale)
        return x + res



def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class Discriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            # nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

        if init_weights:
            self.init_weights('xaiver')

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]



class LocalDiscriminator(BaseNetwork):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(LocalDiscriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )



        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        outputs = conv4
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv4)

        return torch.mean(outputs), [conv1, conv2, conv3, conv4]

class TransmissionEstimator(nn.Module):
    def __init__(self, width=15,):
        super(TransmissionEstimator, self).__init__()
        self.width = width
        self.t_min = 0.2
        self.alpha = 2.5
        self.A_max = 220.0/255
        self.omega=0.95
        self.p = 0.001
        self.max_pool = nn.MaxPool2d(kernel_size=width,stride=1)
        self.max_pool_with_index = nn.MaxPool2d(kernel_size=width, return_indices=True)
        self.guided_filter = GuidedFilter(r=10,eps=1e-3)


    def get_dark_channel(self, x):
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = F.pad(x, (self.width//2, self.width//2,self.width//2, self.width//2), mode='constant', value=1)
        x = -(self.max_pool(-x))
        return x

    def get_atmosphere_light(self,I,dc):
        n,c,h,w = I.shape
        flat_I = I.view(n,c,-1)
        flat_dc = dc.view(n,1,-1)
        searchidx = torch.argsort(flat_dc, dim=2, descending=True)[:,:,:int(h*w*self.p)]
        searchidx = searchidx.expand(-1,3,-1)
        searched = torch.gather(flat_I,dim=2, index=searchidx)
        return torch.max(searched, dim=2 ,keepdim=True)[0].unsqueeze(3)

    def get_transmission(self, I, A):
        return 1-self.omega* self.get_dark_channel(I/A)

    def get_refined_transmission(self, I, rawt):
        I_max = torch.max(I.contiguous().view(I.shape[0],-1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        I_min = torch.min(I.contiguous().view(I.shape[0],-1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        normI = (I - I_min)/(I_max-I_min+0.01)
        refinedT = self.guided_filter(normI, rawt)

        return refinedT

    def get_radiance(self,I, A, t):
        return (I-A)/t + A



    def get_depth(self, I):
        I_dark = self.get_dark_channel(I)

        A = self.get_atmosphere_light(I, I_dark)
        A[A>self.A_max] = self.A_max
        rawT = self.get_transmission(I, A)

        # print(I)

        refinedT = self.get_refined_transmission(I, rawT)
        return refinedT
        
    def direct_get_transmission(self,I):
        I_dark = self.get_dark_channel(I)

        A = self.get_atmosphere_light(I, I_dark).clamp(0.01,1)
        A[A>self.A_max] = self.A_max
        rawT = self.get_transmission(I, A)

        # print(I)

        refinedT = self.get_refined_transmission(I, rawT)
        return refinedT

    def get_atmosphere_light_new(self, I):
        I_dark = self.get_dark_channel(I)
        A = self.get_atmosphere_light(I, I_dark)
        A[A > self.A_max] = self.A_max
        return A



class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        #assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return torch.mean(mean_A * x + mean_b, dim=1 ,keepdim=True)

def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output

class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)



