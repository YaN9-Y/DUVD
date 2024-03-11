import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .networks import Discriminator, LocalDiscriminator , TransmissionEstimator, HazeProduceNet, HazeRemovalNet
from .loss import  AdversarialLoss
import numpy as np
from .utils import warp
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

        if config.MODEL == 1:
            self.name = 'reconstruct'


        self.config = config
        self.iteration = 0

        self.gen_weights_path = os.path.join(config.PATH, 'weights.pth')
        self.gen_optimizer_path = os.path.join(config.PATH, 'optimizer_'+self.name + '.pth')
        self.dis_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.transformer_weights_path = os.path.join(config.PATH, self.name + '.pth')
        self.transformer_discriminator_weights_path = os.path.join(config.PATH, self.name + '_dis.pth')
        self.reconstructor_weights_path = os.path.join(config.PATH, self.name + '.pth')

    def load(self):
        pass

    def save(self, save_best, psnr, iteration):
        pass




class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.adversarial_loss = AdversarialLoss(type='lsgan')
        self._mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).cuda()
        self._std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).cuda()
        self.use_dc_A = True if config.USE_DC_A == 1 else False
        self.transmission_estimator = TransmissionEstimator()
        self.current_q = 0

        self.net_h2c = HazeRemovalNet(base_channel_nums=config.BASE_CHANNEL_NUM, min_beta=config.MIN_BETA, max_beta=config.MAX_BETA, norm_type=config.NORM_TYPE)
        self.net_c2h = HazeProduceNet(base_channel_nums=config.BASE_CHANNEL_NUM, in_channels=3, out_channels=3 , min_beta=config.MIN_BETA, max_beta=config.MAX_BETA, norm_type=config.NORM_TYPE)
        #self.net_depth = torch.hub.load("./intel-isl_MiDaS_master", 'MiDaS_small',source='local').cuda().eval()
        self.net_depth = torch.hub.load("intel-isl/MiDaS", 'MiDaS_small').cuda().eval()

        self.get_random_patch_function = TF.RandomCrop(size=[self.config.CROP_SIZE//2])

        self.min_depth = 0
        self.max_depth = 1685
        self.epoch = 0

        if config.MODE == 1:

            self.discriminator_h2c = Discriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)
            self.discriminator_c2h = Discriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)
            self.discriminator_h2c_local = LocalDiscriminator(in_channels=3, use_spectral_norm=True, use_sigmoid=True)


            self.optimizer = optim.Adam(
                [
                    {'params': self.net_c2h.parameters()},
                    {'params': self.net_h2c.parameters()},
                ],

                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2),
                weight_decay=config.WEIGHT_DECAY
            )


            self.optimizer_dis = optim.Adam(
                [
                    {'params': self.discriminator_h2c.parameters()},
                    {'params': self.discriminator_c2h.parameters()},
                    {'params': self.discriminator_h2c_local.parameters()},
                ],

                lr=float(config.LR * config.D2G_LR),
                betas=(config.BETA1, config.BETA2)
            )


            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.T_MAX, last_epoch=self.epoch-1)



    def forward_h2c(self, hazy_imgs, require_paras=False):
        return self.net_h2c(hazy_imgs, require_paras)




    def get_current_lr(self):
        return self.optimizer.param_groups[0]["lr"]




    def save(self, save_best, psnr, iteration):


        if self.config.MODEL == 1:
            torch.save({
                'net_h2c':self.net_h2c.state_dict(),
                'net_c2h':self.net_c2h.state_dict(),
            },self.gen_weights_path[:-4]+'_'+self.name+'.pth' if not save_best else self.gen_weights_path[
                                                                   :-4] +'_'+self.name+ "_%.2f" % psnr + "_RGB" + "_%d" % iteration + '.pth', _use_new_zipfile_serialization=False)
            torch.save({'discriminator_c2h': self.discriminator_c2h.state_dict(),
                        'discriminator_h2c': self.discriminator_h2c.state_dict(),
                        'discriminator_h2c_local': self.discriminator_h2c_local.state_dict()

                        }, self.gen_weights_path[
                           :-4] + '_' + self.name + '_dis.pth' if not save_best else self.gen_weights_path[
                                                                                     :-4] + '_' + self.name + "_dis_%.2f" % psnr +
                                                                                        "_RGB" + "_%d" % iteration + '.pth')

            torch.save({
                'iteration': self.iteration,
                'epoch': self.epoch,
                'scheduler': self.scheduler.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'optimizer_dis': self.optimizer_dis.state_dict(),

            }, self.gen_optimizer_path if not save_best else self.gen_optimizer_path[
                                                           :-4] + "_%.2f" % psnr + "_RGB" + "_%d" % iteration + '.pth', _use_new_zipfile_serialization=False)


    def load(self):

        if os.path.exists(self.gen_weights_path[:-4] + '_reconstruct' + '.pth'):
            print('Loading %s weights...' % 'reconstruct')

            if torch.cuda.is_available():
                weights = torch.load(self.gen_weights_path[:-4] + '_reconstruct' + '.pth')
            else:
                weights = torch.load(self.gen_weights_path[:-4] + '_reconstruct' + '.pth',
                                     lambda storage, loc: storage)

            self.net_h2c.load_state_dict(weights['net_h2c'])
            self.net_c2h.load_state_dict(weights['net_c2h'])

            print('Loading %s weights...' % 'reconstruct complete!')

        if os.path.exists(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth') and self.config.MODE == 1:
            print('Loading discriminator weights...')

            if torch.cuda.is_available():
                weights = torch.load(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth')
            else:
                weights = torch.load(self.gen_weights_path[:-4] + '_' + self.name + '_dis.pth',
                                     lambda storage, loc: storage)

            self.discriminator_c2h.load_state_dict(weights['discriminator_c2h'])
            self.discriminator_h2c.load_state_dict(weights['discriminator_h2c'])
            self.discriminator_h2c_local.load_state_dict(weights['discriminator_h2c_local'])

        if os.path.exists(self.gen_optimizer_path) and self.config.MODE == 1:
            print('Loading %s optimizer...' % self.name)
            if torch.cuda.is_available():
                data = torch.load(self.gen_optimizer_path)
            else:
                data = torch.load(self.gen_optimizer_path, lambda storage, loc: storage)

            self.optimizer.load_state_dict(data['optimizer'])
            self.scheduler.load_state_dict(data['scheduler'])
            self.epoch = data['epoch']
            self.iteration = data['iteration']
            self.optimizer_dis.load_state_dict(data['optimizer_dis'])

    def backward(self, gen_loss):
        gen_loss.backward()
        self.optimizer.step()


    def update_scheduler(self):
        self.scheduler.step()


    def generate_random_flow_parameters(self):
        if self.config.IS_DEBUG == 1:
            distance_translation_front = self.config.MAX_TRANSLATION_FRONT
            distance_translation_right = self.config.MAX_TRANSLATION_RIGHT
            distance_translation_up = self.config.MAX_TRANSLATION_UP
            angle_rotation_x = self.config.MAX_ROTATION_X
            angle_rotation_y = self.config.MAX_ROTATION_Y
        else:
            distance_translation_front = (2 * np.random.rand()-1) * self.config.MAX_TRANSLATION_FRONT
            distance_translation_right = (2 * np.random.rand()-1) * self.config.MAX_TRANSLATION_RIGHT
            distance_translation_up = (2 * np.random.rand()-1) * self.config.MAX_TRANSLATION_UP
            angle_rotation_x = (2 * np.random.rand()-1) * self.config.MAX_ROTATION_X
            angle_rotation_y = (2 * np.random.rand()-1) * self.config.MAX_ROTATION_Y

        return distance_translation_front, distance_translation_right, distance_translation_up, angle_rotation_x, angle_rotation_y

    def generate_fake_flow(self, depth, distance_front, distance_right, distance_up, angle_x, angle_y
                           ):
        pixel_max_height, pixel_max_width = depth.shape[2:4]

        flow = torch.zeros(1,2,pixel_max_height, pixel_max_width)

        distance_max_height_IP = self.config.DISTANCE_HEIGHT
        distance_max_width_IP = self.config.DISTANCE_WIDTH
        focal_length = self.config.FOCAL_LENGTH


        angle_x = angle_x / 180 * 3.14159
        angle_y = angle_y / 180 * 3.14159

        depth = depth.squeeze(1).squeeze(0).cpu()

        x_coord = torch.linspace(-(pixel_max_width) // 2, (pixel_max_width) // 2, pixel_max_width)
        x_coord = torch.tile(x_coord.reshape(-1, pixel_max_width), (pixel_max_height, 1))

        y_coord = torch.linspace((pixel_max_height) // 2,-(pixel_max_height) // 2, pixel_max_height)
        y_coord = torch.tile(y_coord.reshape(pixel_max_height, -1), (1, pixel_max_width))

        x_distance_IP = x_coord * (distance_max_width_IP / pixel_max_width)
        y_distance_IP = y_coord * (distance_max_height_IP / pixel_max_height)

        xx = torch.zeros(depth.shape)
        yy = torch.zeros(depth.shape)

        xx += distance_front * x_distance_IP / depth
        yy += distance_front * y_distance_IP / depth

        xx -= distance_right * focal_length / depth
        yy -= distance_up * focal_length / depth

        xx += 2.5*(x_distance_IP * y_distance_IP / focal_length * angle_x) - (focal_length * angle_y) - (
                    x_distance_IP ** 2 / focal_length * angle_y)
        yy += (angle_x * focal_length) + (angle_x * y_distance_IP ** 2 / focal_length) - 2.5*(
                    x_distance_IP * y_distance_IP / focal_length * angle_y)

        xx = xx / (distance_max_width_IP) * pixel_max_width
        yy = yy / (distance_max_height_IP) * pixel_max_height

        flow[0,0,:,:] += xx
        flow[0,1,:,:] -= yy   

        return flow.cuda()

    def generate_fake_next_frame(self, first_frame, fake_flow, requires_mask=False): # first_frame (N,3,H,W)

        if first_frame.is_cuda:
            fake_flow = fake_flow.cuda()
        if requires_mask:
            second_frame, mask = warp(first_frame, fake_flow, requires_mask=requires_mask)
            return second_frame, mask
        second_frame = warp(first_frame, fake_flow)


        return second_frame #(N,3,H,W), (N,2,H,W)





    def generate_fake_depth(self,depth, fake_flow, distance_front, distance_right, distance_up, requires_depth_before_warping=False): # depth: (N,1,H,W) flow (N,2,H,W)
        N,_, pixel_height, pixel_width = depth.shape

        depth = depth.reshape(pixel_height, pixel_width).cpu()
        depth_before_warping = depth

        depth_before_warping -= 2.5 * distance_front

        depth_before_warping = depth_before_warping.reshape(N,1,pixel_height,pixel_width)
        fake_flow = fake_flow.reshape(N,2,pixel_height,pixel_width).cpu()

        depth_new = warp(depth_before_warping,fake_flow)

        if requires_depth_before_warping:
            return depth_new.cuda(), depth_before_warping.cuda()

        else:
            return depth_new.cuda() #(N,1,H,W)


    def forward_c2h_given_parameters(self, input_frame, beta, depth, requires_direct_fog=False):
        return self.net_c2h(input_frame,depth.detach(), beta, requires_direct_fog=requires_direct_fog)

    def forward_c2h_random_parameters(self, input_frame, depth, requires_direct_fog=False):
        return self.net_c2h.forward_random_parameters(input_frame,depth.detach(),requires_direct_fog=requires_direct_fog)

    def generate_next_hazy_frame(self, input_frame, depth_input, depth_next, beta_input, flow, requires_mask=False):
        A = self.transmission_estimator.get_atmosphere_light_new(input_frame)
        next_hazy_frame = (((input_frame.detach() - A) * torch.exp(-beta_input * (depth_next.detach() - depth_input.detach()).clamp(min=-1,max=1))) + A).clamp(0, 1)

        if requires_mask:
            next_hazy_frame, mask = (self.generate_fake_next_frame(next_hazy_frame, flow, requires_mask))
            return next_hazy_frame.clamp(0, 1), mask
        next_hazy_frame = (self.generate_fake_next_frame(next_hazy_frame, flow, requires_mask)).clamp(0, 1)

        return next_hazy_frame




    def forward_depth(self, input_frame):
        with torch.no_grad():
            N,_,H,W = input_frame.shape


            raw_depth = self.net_depth(input_frame).reshape(N,1,H,W)

            normalize_depth = (raw_depth - self.min_depth) / (self.max_depth - self.min_depth + 0.1)
            normalize_depth = 1 - normalize_depth

            depth = self.config.MIN_DEPTH + normalize_depth * (
                    self.config.MAX_DEPTH - self.config.MIN_DEPTH)

            f_max = torch.max(input_frame.contiguous().view(input_frame.shape[0], -1), dim=1, keepdim=True)[
                0].unsqueeze(
                2).unsqueeze(3)
            f_min = torch.min(input_frame.contiguous().view(input_frame.shape[0], -1), dim=1, keepdim=True)[
                0].unsqueeze(
                2).unsqueeze(3)
            norm_frame = (input_frame - f_min) / (f_max - f_min + 0.1)

            depth = self.transmission_estimator.guided_filter(norm_frame, depth)


            return depth.reshape(N,1,H,W)



    def process(self, clean_images, hazy_images):
        self.iteration += 1
        self.current_q = min(float(self.epoch) / 50,1) * 0.5

        self.optimizer_dis.zero_grad()
        self.discriminator_h2c.zero_grad()
        self.discriminator_c2h.zero_grad()
        self.discriminator_h2c_local.zero_grad()


        clean_images_h2c, beta_pred_h2c = self.forward_h2c(hazy_images,require_paras=True)
        depth_h2c = self.forward_depth(clean_images_h2c).detach()
        hazy_images_h2c2h = self.forward_c2h_given_parameters(clean_images_h2c,depth_h2c.detach(),beta_pred_h2c.detach())


        depth_c = self.forward_depth(clean_images).detach()
        hazy_images_c2h, beta_c2h = self.forward_c2h_random_parameters(clean_images, depth_c)
        clean_images_c2h2c, beta_c2h2c_pred = self.forward_h2c(hazy_images_c2h,require_paras=True)

        # ---- generate synthetic next clean frame -----
        distance_translation_front_c, distance_translation_right_c, distance_translation_up_c, angle_rotation_x_c, angle_rotation_y_c = self.generate_random_flow_parameters()
        next_frame_synthetic_flow_c = self.generate_fake_flow(depth_c, distance_translation_front_c, distance_translation_right_c, distance_translation_up_c, angle_rotation_x_c, angle_rotation_y_c)
        depth_c_next_frame,depth_c_next_frame_before_warping = self.generate_fake_depth(depth_c, next_frame_synthetic_flow_c, distance_translation_front_c, distance_translation_right_c, distance_translation_up_c, requires_depth_before_warping=True)
        # # # \tilde{x}_{t+1}
        clean_image_next_frame_synthetic, mask_clean_next_frame_synthetic = self.generate_fake_next_frame(clean_images,next_frame_synthetic_flow_c, requires_mask=True)

        # # # W(\hat{x^t}, \tilde{f})
        hazy_images_c2h_next_frame = self.generate_next_hazy_frame(hazy_images_c2h.detach(),depth_c.detach(), depth_c_next_frame_before_warping.detach(),beta_input=beta_c2h.detach(), flow=next_frame_synthetic_flow_c.detach() )
        # #
        # # # G_X(W(\hat{x^t}, \tilde{f}))
        clean_images_c2h2c_next_frame = self.forward_h2c(hazy_images_c2h_next_frame)
        # #
        # # # G_Y(\tilde{x}_{t+1})
        hazy_images_c2h_next_frame_from_synthetic = self.forward_c2h_given_parameters(clean_image_next_frame_synthetic, beta_c2h.detach(), depth_c_next_frame.detach())
        # #
        # # ---- generate next synthetic hazy frame ----
        distance_translation_front_h, distance_translation_right_h, distance_translation_up_h, angle_rotation_x_h, angle_rotation_y_h = self.generate_random_flow_parameters()
        next_frame_synthetic_flow_h = self.generate_fake_flow(depth_h2c, distance_translation_front_h, distance_translation_right_h, distance_translation_up_h, angle_rotation_x_h, angle_rotation_y_h)
        depth_h2c_next_frame,fake_depth_h2c_next_frame_before_warping = self.generate_fake_depth(depth_h2c.detach(),next_frame_synthetic_flow_h, distance_translation_front_h, distance_translation_right_h, distance_translation_up_h, requires_depth_before_warping=True)
        # #
        # # # \tilde{y}_{t+1}
        hazy_image_next_frame_synthetic, mask_hazy_next_frame_synthetic = self.generate_next_hazy_frame(hazy_images, depth_h2c.detach(), fake_depth_h2c_next_frame_before_warping, beta_pred_h2c, next_frame_synthetic_flow_h, requires_mask=True)
        # # # W(\hat{y^t}, \tilde{f})
        # #
        clean_images_h2c_next_frame = self.generate_fake_next_frame(clean_images_h2c, next_frame_synthetic_flow_h)
        # #
        # # # G_Y(W(\hat{y^t}, \tilde{f}))
        # #
        hazy_images_h2c2h_next_frame = self.forward_c2h_given_parameters(clean_images_h2c_next_frame,beta_pred_h2c.detach(),depth_h2c_next_frame.detach())
        # #
        # # # G_X(\tilde{y}_{t+1})
        clean_images_h2c_next_frame_from_synthetic = self.forward_h2c(hazy_image_next_frame_synthetic)
        # #

        gen_loss = 0
        dis_loss = 0

        dis_real_clean, _ = self.discriminator_h2c(clean_images)
        dis_fake_clean, _ = self.discriminator_h2c(
            clean_images_h2c.detach())

        dis_clean_real_loss = self.adversarial_loss((dis_real_clean), is_real=True, is_disc=True)
        dis_clean_fake_loss = self.adversarial_loss((dis_fake_clean), is_real=False, is_disc=True)

        dis_clean_loss = (dis_clean_real_loss + dis_clean_fake_loss) / 2
        dis_clean_loss.backward()

        dis_real_haze, _ = self.discriminator_c2h(
            (hazy_images))
        dis_fake_haze, _ = self.discriminator_c2h(
            hazy_images_c2h.detach())

        dis_haze_real_loss = self.adversarial_loss((dis_real_haze), is_real=True, is_disc=True)
        dis_haze_fake_loss = self.adversarial_loss((dis_fake_haze), is_real=False, is_disc=True)
        dis_haze_loss = (dis_haze_real_loss + dis_haze_fake_loss) / 2
        dis_haze_loss.backward()

        clean_images_patch = self.get_random_patch(clean_images)
        clean_images_h2c_patch = self.get_random_patch_according_depth(clean_images_h2c, depth_h2c.detach(),self.current_q)

        dis_real_clean_local, _ = self.discriminator_h2c_local(clean_images_patch)
        dis_fake_clean_local, _ = self.discriminator_h2c_local(
            clean_images_h2c_patch.detach())

        dis_clean_real_loss_local = self.adversarial_loss((dis_real_clean_local), is_real=True, is_disc=True)
        dis_clean_fake_loss_local = self.adversarial_loss((dis_fake_clean_local), is_real=False, is_disc=True)

        dis_clean_loss_local = (dis_clean_fake_loss_local + dis_clean_real_loss_local) / 2
        dis_clean_loss_local.backward()

        dis_loss += (dis_clean_fake_loss + dis_clean_real_loss + dis_haze_real_loss + dis_haze_fake_loss) / 4

        self.optimizer_dis.step()
        self.optimizer.zero_grad()
        self.net_h2c.zero_grad()
        self.net_c2h.zero_grad()

        cycle_loss_c2h2c = self.l1_loss(clean_images,
                                        clean_images_c2h2c)
        cycle_loss_h2c2h = self.l1_loss(hazy_images, hazy_images_h2c2h)
        cycle_loss = (cycle_loss_c2h2c + cycle_loss_h2c2h) / 2

        mask_clean_next_frame_synthetic = mask_clean_next_frame_synthetic.detach()
        mask_hazy_next_frame_synthetic = mask_hazy_next_frame_synthetic.detach()

        recycle_loss_clean_side = self.l1_loss(clean_image_next_frame_synthetic*mask_clean_next_frame_synthetic, clean_images_c2h2c_next_frame*mask_clean_next_frame_synthetic)
        recycle_loss_hazy_side = self.l1_loss(hazy_image_next_frame_synthetic*mask_hazy_next_frame_synthetic, hazy_images_h2c2h_next_frame* mask_hazy_next_frame_synthetic)
        recycle_loss = (recycle_loss_clean_side + recycle_loss_hazy_side) / 2

        spatial_loss_clean_side = self.l1_loss(hazy_images_c2h_next_frame_from_synthetic*mask_clean_next_frame_synthetic,hazy_images_c2h_next_frame*mask_clean_next_frame_synthetic)
        spatial_loss_hazy_side = self.l1_loss(clean_images_h2c_next_frame_from_synthetic*mask_hazy_next_frame_synthetic, clean_images_h2c_next_frame*mask_hazy_next_frame_synthetic)
        spatial_loss = (spatial_loss_hazy_side+spatial_loss_clean_side)/2

        beta_loss = self.l1_loss(beta_c2h2c_pred, beta_c2h)

        gen_fake_haze, _ = self.discriminator_c2h(
            (hazy_images_c2h))
        gen_fake_clean, _ = self.discriminator_h2c(
            clean_images_h2c)
        gen_fake_clean_local, _ = self.discriminator_h2c_local(
            clean_images_h2c_patch)

        gen_fake_haze_ganloss = self.adversarial_loss((gen_fake_haze), is_real=True, is_disc=False)
        gen_fake_clean_ganloss = self.adversarial_loss((gen_fake_clean), is_real=True, is_disc=False)
        gen_fake_clean_local_ganloss = self.adversarial_loss((gen_fake_clean_local), is_real=True, is_disc=False)

        gen_gan_loss = (gen_fake_clean_ganloss + gen_fake_haze_ganloss + gen_fake_clean_local_ganloss) / 2


        gen_loss += self.config.GAN_LOSS_WEIGHT * gen_gan_loss
        gen_loss += self.config.CYCLE_LOSS_WEIGHT * cycle_loss
        gen_loss += self.config.RECYCLE_LOSS_WEIGHT * recycle_loss
        gen_loss += self.config.SPATIAL_LOSS_WEIGHT * spatial_loss
        gen_loss += self.config.BETA_LOSS_WEIGHT * beta_loss

        gen_loss.backward()
        nn.utils.clip_grad_value_(parameters=self.net_c2h.parameters(), clip_value=0.5)
        nn.utils.clip_grad_value_(parameters=self.net_h2c.parameters(), clip_value=0.5)
        self.optimizer.step()

        logs = [
            ("g_total", gen_loss.item()),
            ("d_dis", dis_loss.item()),
        ]
        return clean_images_c2h2c, gen_loss, dis_loss, logs


    def get_random_patch(self, images):
        images = TFF.pad(images, padding=self.config.CROP_SIZE // 4, padding_mode='reflect')
        patch = self.get_random_patch_function(images)
        return patch


    def get_random_patch_according_depth(self, images, depths, q):
        n,c,h,w = depths.shape
        top_index = h*w
        threshold = (int)(top_index*q)
        random_index = random.randint(threshold,top_index-1)
        
        sorted_depths, sorted_indexs = torch.sort(depths.view(-1,1,h*w))
        index_choice = sorted_indexs[0,0,random_index].item()
        h_choice = index_choice // h
        w_choice = index_choice % w
        images = TFF.pad(images, padding=self.config.CROP_SIZE // 4, padding_mode='reflect')
        patch = TFF.crop(images,h_choice,w_choice,self.config.CROP_SIZE // 2,self.config.CROP_SIZE // 2)
        
        return patch









