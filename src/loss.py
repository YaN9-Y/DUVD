import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.autograd as autograd
from functools import reduce
from math import exp
from .utils import rgb2hsv
from torch.autograd import Variable
from torch.nn.modules.utils import _triple, _pair, _single
from kornia.color import rgb_to_hsv


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]




class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        #self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [0, 0.4, 0.6, 0, 1.0]
        self.ab = ablation

    # def forward(self, a, p, n):
    #     a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
    #     loss = 0
    #
    #     d_ap, d_an = 0, 0
    #     for i in range(len(a_vgg)):
    #         d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
    #         if not self.ab:
    #             d_an = self.l1(a_vgg[i], n_vgg[i].detach())
    #             contrastive = d_ap / (d_an + 1e-7)
    #         else:
    #             contrastive = d_ap
    #
    #         loss += self.weights[i] * contrastive
    #     return loss
    #
    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i])
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i])
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss




class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss




class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss

class DecoderStyleLoss(nn.Module):
    def __init__(self):
        super(DecoderStyleLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y, masks):
        assert len(x) == len(y)
        style_loss = 0.

        for i in range(len(x)):
            H,W = x[i].shape[2:4]
            scaled_masks = torch.nn.functional.interpolate(masks,[H,W])
            style_loss += self.criterion(self.compute_gram(x[i]*scaled_masks),self.compute_gram(y[i]*scaled_masks))

        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])


        return content_loss



class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            #'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            #'relu3_2': relu3_2,
            #'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            #'relu4_2': relu4_2,
            #'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            #'relu5_3': relu5_3,
            #'relu5_4': relu5_4,
        }
        return out


class SimilarityContrastLoss(nn.Module):
    def __init__(self):
        super(SimilarityContrastLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [0, 0.4, 0.6, 0, 1.0]

    def compute_similarity(self, x, y):
        b, ch, h, w = x.size()
        x = x.view(b, ch, w * h)
        y = y.view(b, ch, w * h)
        y_T = y.transpose(1, 2)
        G = x.bmm(y_T) / (h * w * ch)
        self.weights = [0, 0.4, 0.6, 0, 1.0]

        return G

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)

        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = torch.mean(torch.abs(self.compute_similarity(a_vgg[i], p_vgg[i])))
            # if not self.ab:
            #     d_an = self.l1(a_vgg[i], n_vgg[i])
            #     contrastive = d_ap / (d_an + 1e-7)
            # else:
            #     contrastive = d_ap
            d_an = torch.mean(torch.abs(self.compute_similarity(a_vgg[i], n_vgg[i])))
            contrastive = d_an / ( d_ap + 1e-7)


            loss += self.weights[i] * contrastive
        return loss

# class VGG19(nn.Module):
#     def __init__(self, pool='max'):
#         super(VGG19, self).__init__()
#         self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
#         self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#         self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         if pool == 'max':
#             self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
#             self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
#         elif pool == 'avg':
#             self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
#             self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
#
#     def forward(self, x):
#         out = {}
#         out['r11'] = F.relu(self.conv1_1(x))
#         out['r12'] = F.relu(self.conv1_2(out['r11']))
#         out['p1'] = self.pool1(out['r12'])
#         out['r21'] = F.relu(self.conv2_1(out['p1']))
#         out['r22'] = F.relu(self.conv2_2(out['r21']))
#         out['p2'] = self.pool2(out['r22'])
#         out['r31'] = F.relu(self.conv3_1(out['p2']))
#         out['r32'] = F.relu(self.conv3_2(out['r31']))
#         out['r33'] = F.relu(self.conv3_3(out['r32']))
#         out['r34'] = F.relu(self.conv3_4(out['r33']))
#         out['p3'] = self.pool3(out['r34'])
#         out['r41'] = F.relu(self.conv4_1(out['p3']))
#         out['r42'] = F.relu(self.conv4_2(out['r41']))
#         out['r43'] = F.relu(self.conv4_3(out['r42']))
#         out['r44'] = F.relu(self.conv4_4(out['r43']))
#         out['p4'] = self.pool4(out['r44'])
#         out['r51'] = F.relu(self.conv5_1(out['p4']))
#         out['r52'] = F.relu(self.conv5_2(out['r51']))
#         out['r53'] = F.relu(self.conv5_3(out['r52']))
#         out['r54'] = F.relu(self.conv5_4(out['r53']))
#         out['p5'] = self.pool5(out['r54'])
#         return out
#


class LaplacianLoss(nn.Module):
    def __init__(self):
        super(LaplacianLoss, self).__init__()
        self.weight = torch.tensor([[[[0,1,0],[1, -4 ,1], [0 ,1 ,0]]]]).float().cuda().expand(1,256,3,3)
        self.criterion = nn.L1Loss()
    def __call__(self, x, y):
        return self.criterion(F.conv2d(x,self.weight,padding=1), F.conv2d(y,self.weight,padding=1)) / 256


class DecoderFeatureLoss(nn.Module):
    def __init__(self):
        super(DecoderFeatureLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def __call__(self, x, y, weight=[1.0,1.0,1.0,1.0,1.0]): #convconv64_256, deconv64_128_128,conv128_128,deconv128_256_64,conv256_3
        assert len(weight) == 5 and len(x) == len(y)
        decoder_feature_loss = 0.
        for i in range(len(x)):
            decoder_feature_loss += weight[i] * self.criterion(x[i],y[i])

        return decoder_feature_loss



############################### DMFN Part ####################################3

class WGANLoss(nn.Module):
    def __init__(self):
        super(WGANLoss, self).__init__()

    def __call__(self, input, target):
        d_loss = (input - target).mean()
        g_loss = -input.mean()
        return {'g_loss': g_loss, 'd_loss': d_loss}


def gradient_penalty(xin, yout, mask=None):
    gradients = autograd.grad(yout, xin, create_graph=True,
                              grad_outputs=torch.ones(yout.size()).cuda(), retain_graph=True, only_inputs=True)[0]
    if mask is not None:
        gradients = gradients * mask
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def random_interpolate(gt, pred):
    batch_size = gt.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).cuda()
    # alpha = alpha.expand(gt.size()).cuda()
    interpolated = gt * alpha + pred * (1 - alpha)
    return interpolated

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.featlayer = VGG19FeatLayer()
        for k, v in self.featlayer.named_parameters():
            v.requires_grad = False
        self.self_guided_layers = ['relu1_1', 'relu2_1']
        self.feat_vgg_layers = ['relu{}_1'.format(x + 1) for x in range(5)]
        self.lambda_loss = 25
        self.gamma_loss = 1
        self.align_loss, self.guided_loss, self.fm_vgg_loss = None, None, None
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.coord_y, self.coord_x = torch.meshgrid(torch.arange(-1, 1, 1 / 16), torch.arange(-1, 1, 1 / 16))
        self.coord_y, self.coord_x = self.coord_y.cuda(), self.coord_x.cuda()

    def sum_normalize(self, featmaps):
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        return featmaps / reduce_sum

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-5
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        relative_dist = cdist / (div + epsilon)
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist) / self.nn_stretch_sigma)
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def calc_align_loss(self, gen, tar):
        def sum_u_v(x):
            area = x.shape[-2] * x.shape[-1]
            return torch.sum(x.view(-1, area), -1) + 1e-7

        sum_gen = sum_u_v(gen)
        sum_tar = sum_u_v(tar)
        c_u_k = sum_u_v(self.coord_x * tar) / sum_tar
        c_v_k = sum_u_v(self.coord_y * tar) / sum_tar
        c_u_k_p = sum_u_v(self.coord_x * gen) / sum_gen
        c_v_k_p = sum_u_v(self.coord_y * gen) / sum_gen
        out = F.mse_loss(torch.stack([c_u_k, c_v_k], -1), torch.stack([c_u_k_p, c_v_k_p], -1), reduction='mean')
        return out

    def forward(self, gen, tar, mask_guidance, weight_fn):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)

        guided_loss_list = []
        mask_guidance = mask_guidance.unsqueeze(1)
        for layer in self.self_guided_layers:
            guided_loss_list += [F.l1_loss(gen_vgg_feats[layer] * mask_guidance, tar_vgg_feats[layer] * mask_guidance, reduction='sum') * weight_fn(tar_vgg_feats[layer])]
            mask_guidance = self.avg_pool(mask_guidance)
        self.guided_loss = reduce(lambda x, y: x + y, guided_loss_list)

        content_loss_list = [F.l1_loss(gen_vgg_feats[layer], tar_vgg_feats[layer], reduction='sum') * weight_fn(tar_vgg_feats[layer]) for layer in self.feat_vgg_layers]
        self.fm_vgg_loss = reduce(lambda x, y: x + y, content_loss_list)

        self.align_loss = self.calc_align_loss(gen_vgg_feats['relu4_1'], tar_vgg_feats['relu4_1'])

        return self.align_loss, self.guided_loss, self.fm_vgg_loss
        return self.gamma_loss * self.align_loss + self.lambda_loss * (self.guided_loss + self.fm_vgg_loss)



class VGG19FeatLayer(nn.Module):
    def __init__(self):
        super(VGG19FeatLayer, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True).features.eval().cuda()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()

    def forward(self, x):
        out = {}
        x = x - self.mean
        ci = 1
        ri = 0
        for layer in self.vgg19.children():
            if isinstance(layer, nn.ReLU):
                ri += 1
                name = 'relu{}_{}'.format(ci, ri)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                ri = 0
                name = 'pool_{}'.format(ci)
                ci += 1
            x = layer(x)
            if isinstance(layer, nn.ReLU) and ri == 1:
                out[name] = x
                if ci == 5:
                    break
        # print([x for x in out])
        return out

class RaganLoss(nn.Module):
    def __init__(self, config):
        super(RaganLoss, self).__init__()
        self.BCELoss = nn.BCEWithLogitsLoss().to(config.DEVICE)
        self.zeros = torch.zeros((config.BATCH_SIZE, 1)).to(config.DEVICE)
        self.ones = torch.ones((config.BATCH_SIZE, 1)).to(config.DEVICE)

    def Dra(self, x1, x2):
        return x1 - torch.mean(x2)

    def forward(self, x_real, x_fake, type):
        assert type in ['adv', 'dis']
        if type == 'dis':
            return (self.BCELoss(self.Dra(x_real, x_fake), self.ones) + self.BCELoss(self.Dra(x_fake, x_real), self.zeros)) / 2
        else:
            return (self.BCELoss(self.Dra(x_real, x_fake), self.zeros) + self.BCELoss(self.Dra(x_fake, x_real), self.ones)) / 2



#############  ssim loss ##########33
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

## change to no L:
    # C1 = (0.01 * L) ** 2
    # C2 = (0.03 * L) ** 2

    C1 = (0.01 ) ** 2
    C2 = (0.03 ) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=None):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)

        # Relu normalize (not compliant with original definition)
        if normalize == "relu":
            ssims.append(torch.relu(sim))
            mcs.append(torch.relu(cs))
        else:
            ssims.append(sim)
            mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    # Simple normalize (not compliant with original definition)
    # TODO: remove support for normalize == True (kept for backward support)
    if normalize == "simple" or normalize == True:
        ssims = (ssims + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

    def __call__(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss,self).__init__()

    def forward(self, x):
        height = x.size()[2]
        width = x.size()[3]
        tv_h = torch.div(torch.sum(torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])),(x.size()[0]*x.size()[1]*(height-1)*width))
        tv_w = torch.div(torch.sum(torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])),(x.size()[0]*x.size()[1]*(height)*(width-1)))
        return tv_w + tv_h
    

class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def forward(self, f_pred, f_gt):
        weights = self.cal_weight(f_gt)
        loss = torch.mean(torch.sum(weights*(torch.mean(torch.mean(torch.abs(f_pred-f_gt), dim=3, keepdim=True),dim=2,keepdim=True)),dim=1))

        #loss = torch.mean(torch.abs(weights*(f_pred-f_gt)))

        return loss




    def cal_weight(self, f):
        f_mean = torch.mean(f, dim=1, keepdim=True)
        sigma = torch.sqrt(torch.sum(torch.sum((f - f_mean) ** 2, dim=3, keepdim=True), dim=2, keepdim=True))
        r = torch.sum(torch.sum(f - f_mean, dim=3, keepdim=True), dim=2, keepdim=True) / sigma

        weights = torch.nn.functional.softmax(r).detach()

        return weights

class HazeReconstructionLoss(nn.Module):
    def __init__(self):
        super(HazeReconstructionLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred_transmission, pred_features, hazy_features, hazy_imgs):
        A = self.get_A(hazy_features)
        reconstructed_hazy_features = pred_features * pred_transmission + A * (1-pred_transmission)
        return self.criterion(reconstructed_hazy_features, hazy_features)


    def get_A(self, hazy_features):
        n,c,h,w = hazy_features.shape
        dark_channel = torch.min(hazy_features, dim=1)[0]
        A_index_unfold = torch.max(dark_channel.reshape(n,h*w),dim=1)[1]
        # print(A_index_unfold.shape)
        # print(A_index_unfold.unsqueeze(0).transpose(1, 0).unsqueeze(1).repeat(1, c, 1).shape)
        A = torch.gather(hazy_features.reshape(n,c,h*w), 2, A_index_unfold.unsqueeze(0).transpose(1,0).unsqueeze(1).repeat(1,c,1)).unsqueeze(3)

        # print(A.shape)
        return A


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x

class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).cuda()
        self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).cuda()

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding = 1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding = 1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding = 1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding = 1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding = 1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding = 1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.criterion = nn.L1Loss()
        self.grad_caculator = Get_gradient_nopadding()

    def forward(self, img1, img2):
        grad1 = self.grad_caculator(img1)
        grad2 = self.grad_caculator(img2)

        return self.criterion(grad1, grad2)

class HueLoss(nn.Module):
    def __init__(self):
        super(HueLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, img_h1, img_h2):
        hueloss = 180-torch.abs(180-torch.abs(img_h1-img_h2))
        hueloss = torch.mean(hueloss) / 360
        return hueloss

class OnewayHueLoss(nn.Module):
    def __init__(self):
        super(OnewayHueLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, clean_imgs, hazy_imgs):
        n,c,h,w = clean_imgs.shape
        #dark_channel = torch.min(hazy_imgs, dim=1, keepdim=True)[0]
        clean_imgs_hue = rgb2hsv(clean_imgs.reshape(n,-1,h,w))[:,0].unsqueeze(1)
        hazy_imgs_hue = rgb2hsv(hazy_imgs.reshape(n,-1,h,w))[:,0].unsqueeze(1)

        hueloss = torch.mean((180-torch.abs(180-torch.abs(clean_imgs_hue-hazy_imgs_hue)))/360)
        #hueloss = torch.mean(hueloss * (1-dark_channel)) / 360
        return hueloss


class DepthAwareOnewayHueLoss(nn.Module):    #######输入的depth是近1远0
    def __init__(self):
        super(DepthAwareOnewayHueLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self,clean_imgs, hazy_imgs, depth):
        clean_imgs_hue = rgb2hsv(clean_imgs)[:, 0].unsqueeze(1)
        hazy_imgs_hue = rgb2hsv(hazy_imgs)[:, 0].unsqueeze(1)

        hueloss = 180 - torch.abs(180 - torch.abs(clean_imgs_hue - hazy_imgs_hue))
        hueloss = torch.mean(hueloss * (0.8*depth + 0.2)) / 360
        return hueloss


#Both rgb2hsv and hsv2rgb take images in a mini-batch of images (NCHW format) as input.
# The range of H (hue) is from 0 to 360 (the outside of the range loops). The range of RGB and SV is 0 to 1.

class AtmosphereHomogenousLoss(nn.Module):
    def __init__(self):
        super(AtmosphereHomogenousLoss, self).__init__()

    def forward(self, A):
        n,c,h,w = A.shape
        std = torch.std(A.view(n,-1), dim=1)
        return torch.mean(std)


class Relation_DSV(nn.Module):
    def __init__(self):
        super(Relation_DSV, self).__init__()

    def forward(self, gen_clean_imgs, depth):
        clean_imgs_hsv = rgb2hsv(gen_clean_imgs)
        clean_imgs_sv = (clean_imgs_hsv[:,2]-clean_imgs_hsv[:,1]).unsqueeze(1)

        n,c,h,w = gen_clean_imgs.shape

        E_XY = torch.mean((clean_imgs_sv * depth).reshape(n,-1), dim=1, keepdim=True)
        EX = torch.mean((clean_imgs_sv).reshape(n,-1), dim=1, keepdim=True)
        EY = torch.mean((depth).reshape(n,-1), dim=1, keepdim=True)
        DX = torch.var(clean_imgs_sv.reshape(n,-1),dim=1,keepdim=True)
        DY = torch.var(depth.reshape(n,-1),dim=1,keepdim=True)

        corr = torch.abs((E_XY - EX*EY)/(torch.sqrt(DX) * torch.sqrt(DY)))

        return torch.mean(corr)

class GrayWorldLoss(nn.Module):
    def __init__(self):
        super(GrayWorldLoss, self).__init__()

    def forward(self, clean_images):
        n,c,h,w = clean_images.shape
        clean_images = clean_images.reshape(n,c,-1)
        clean_images = torch.mean(clean_images, dim=2)

        loss = torch.abs(clean_images - clean_images[:,[1,2,0]])



        # loss = torch.mean(torch.abs(torch.mean(clean_images,dim=2,keepdim=True)-0.5))
        # return loss
        return torch.mean(loss)



class DarkChannelLoss(nn.Module):
    def __init__(self):
        super(DarkChannelLoss, self).__init__()

    def forward(self, pred_img):
        return torch.mean(torch.abs(self.get_dark_channel(pred_img)))


    def get_dark_channel(input, pred_img):
        channel_wise_min = torch.min(pred_img, keepdim=True, dim=1)[0]
        neighbour_min = -soft_pool2d((-channel_wise_min),kernel_size=15)
        # return torch.mean(torch.exp(neighbour_min,2))
        return neighbour_min


# class DarkChannelLoss(nn.Module):
#     def __init__(self):
#         super(DarkChannelLoss, self).__init__()
#
#     def forward(self, img, patch_size=35):
#         """
#         calculating dark channel of image, the image shape is of N*C*W*H
#         """
#         maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size // 2, patch_size // 2))
#         dc = maxpool(0 - img[:, None, :, :, :])
#
#         target = Variable(torch.FloatTensor(dc.shape).zero_().cuda())
#
#         loss = nn.L1Loss(size_average=True)(-dc, target)
#         return loss
# #
# class TransmissionPenalty(nn.Module):
#     def __init__(self):
#         super(TransmissionPenalty, self).__init__()
#
#     def forward(self, x):
#         n,c,h,w = x.shape
#         return -torch.mean(torch.std(x.view(n,-1),dim=1))


class TransmissionLoss(nn.Module):
    def __init__(self):
        super(TransmissionLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, pred_beta, pred_d, gt_t):
        pred_t = torch.exp(-pred_beta*pred_d)
        # print('----------------------')
        # print(pred_t)
        # print(gt_t)
        # print('-----------------------')
        return self.criterion(pred_t, gt_t)


def soft_pool2d(x, kernel_size=2, stride=None, force_inplace=False):
    kernel_size = _pair(kernel_size)
    if stride is None:
        stride = kernel_size
    else:
        stride = _pair(stride)
    # Get input sizes
    _, c, h, w = x.size()
    # Create per-element exponential value sum : Tensor [b x 1 x h x w]
    e_x = torch.sum(torch.exp(x),dim=1,keepdim=True)
    # Apply mask to input and pool and calculate the exponential sum
    # Tensor: [b x c x h x w] -> [b x c x h' x w']
    return F.avg_pool2d(x.mul(e_x), kernel_size, stride=stride).mul_(sum(kernel_size)).div_(F.avg_pool2d(e_x, kernel_size, stride=stride).mul_(sum(kernel_size)))

class VerticalDepthLoss(nn.Module):
    def __init__(self):
        super(VerticalDepthLoss, self).__init__()

    def forward(self, depth):
        n,c,h,w = depth.shape
        grad_vertical = depth[:,:,1:] - depth[:,:,:h-1]
        return torch.mean(torch.exp(grad_vertical))
