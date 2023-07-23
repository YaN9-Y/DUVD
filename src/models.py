import os
import torch
import torch.nn as nn
import torch.optim as optim
from .networks import Discriminator, LocalDiscriminator , DCPUtils, HazeProduceNet, HazeRemovalNet2
from .loss import  AdversarialLoss
import torchvision.transforms as TF


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()

        if config.MODEL == 1:
            self.name = 'reconstruct'
        elif config.MODEL == 2:
            self.name = 'feature_process'

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
        self.transmission_estimator = DCPUtils()
        self.current_q = 0

        self.net_h2c = HazeRemovalNet2(base_channel_nums=config.BASE_CHANNEL_NUM,min_beta=config.MIN_BETA, max_beta=config.MAX_BETA, norm_type=config.NORM_TYPE)
        self.net_c2h = HazeProduceNet(base_channel_nums=config.BASE_CHANNEL_NUM, in_channels=3, out_channels=3 , min_beta=config.MIN_BETA, max_beta=config.MAX_BETA, norm_type=config.NORM_TYPE)
        #self.net_depth = torch.hub.load("./intel-isl_MiDaS_master", 'MiDaS_small',source='local').cuda().eval()

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
        # if requires_paras: return clean, ed, beta
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


    def forward_c2h_given_parameters(self, input_frame, beta, depth, requires_direct_fog=False):
        return self.net_c2h(input_frame,depth.detach(), beta, requires_direct_fog=requires_direct_fog)

    def forward_c2h_random_parameters(self, input_frame, depth, requires_direct_fog=False):
        return self.net_c2h.forward_random_parameters(input_frame,depth.detach(),requires_direct_fog=requires_direct_fog)





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











