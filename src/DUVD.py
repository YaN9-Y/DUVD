import os
import numpy as np
import torch
import torch.nn.functional as F
import kornia
import cv2
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import Model
from .utils import Progbar, create_dir, imsave
from .metrics import PSNR_RGB
from torch.utils.tensorboard import SummaryWriter
import math


class DUVD():
    def __init__(self, config):
        self.config = config

        self.model = Model(config).to(config.DEVICE)

        self.psnr = PSNR_RGB(255.0).to(config.DEVICE)

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(config, crop_size=None, hazy_flist=config.TEST_HAZY_FLIST,
                                        clean_flist=config.TEST_CLEAN_FLIST, augment=False,
                                        split=self.config.TEST_MODE)

        else:
            self.train_dataset = Dataset(config, crop_size=config.CROP_SIZE, clean_flist=config.TRAIN_CLEAN_FLIST,
                                         hazy_flist=config.TRAIN_HAZY_FLIST, augment=False, split='unpair')
            self.val_dataset = Dataset(config, crop_size=None, hazy_flist=config.VAL_HAZY_FLIST,
                                       clean_flist=config.VAL_CLEAN_FLIST, augment=False, split='pair_test')
            self.test_dataset = Dataset(config, crop_size=None, hazy_flist=config.TEST_HAZY_FLIST,
                                        clean_flist=config.TEST_CLEAN_FLIST, augment=False, split=self.config.TEST_MODE)
            self.sample_dataset = Dataset(config, crop_size=config.CROP_SIZE, clean_flist=config.TRAIN_CLEAN_FLIST,
                                          hazy_flist=config.TRAIN_HAZY_FLIST, augment=False, split='unpair')

            self.sample_iterator = self.sample_dataset.create_iterator(config.SAMPLE_SIZE, shuffle=True)

        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        self.eval_path = os.path.join(config.PATH, 'eval')
        self.log_path = os.path.join(config.PATH, 'logs')

        # if config.RESULTS is not None:
        #     self.results_path = os.path.join(config.RESULTS)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + self.model.name + '.dat')

    def load(self):
        self.model.load()


    def test(self):
        model = self.config.MODEL
        self.model.eval()
        create_dir(self.results_path)
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0

        psnrs = []
        times = []

        with torch.no_grad():
            for items in test_loader:
                if self.test_dataset.split == 'hazy':

                    name = self.test_dataset.load_name(index)[:-4] + '.png'

                    hazy_images = items.to(self.config.DEVICE)
                    index += 1

                    if model == 1:
                        torch.cuda.empty_cache()
                        ## check if the input size is multiple of 4
                        h, w = hazy_images.shape[2:4]
                        print(hazy_images.shape)
                        if h * w > 1000 * 1000:
                            continue
                        hazy_input_images = self.pad_input(hazy_images)
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                        predicted_results = self.model.forward_h2c(hazy_input_images)
                        end.record()
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end))
                        predicted_results = self.crop_result(predicted_results, h, w)
                        predicted_results = self.postprocess(predicted_results)[0]

                        path = os.path.join(self.results_path, self.model.name)
                        create_dir(path)
                        save_name = os.path.join(path, name)
                        imsave(predicted_results, save_name)
                        print(save_name)


            print('AVG times:' + str(np.mean(times)))
            print('Total PSNR_' + ('YCbCr:' if self.config.PSNR == 'YCbCr' else 'RGB:'), np.mean(psnrs))
            print('\nEnd test....')



    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def cuda_list(self, args):
        return [item.to(self.config.DEVICE) for item in args]

    def lr_schedule_cosdecay(self, t, T, init_lr):
        lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
        return lr

    def postprocess(self, img, size=None):
        # [0, 1] => [0, 255]
        if size is not None:
            img = torch.nn.functional.interpolate(img, size, mode='bicubic')
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def generate_color_map(self, imgs, size=[256, 256]):
        # N 1 H W -> N H W 3 color map
        if torch.max(imgs) > 1 or torch.min(imgs) < 0:
            imgs = self.minmax_depth(imgs, blur=True)
        imgs = (imgs * 255.0).int().squeeze(1).cpu().numpy().astype(np.uint8)
        N, height, width = imgs.shape

        colormaps = np.full((N, size[0], size[1], 3), 1)

        for i in range(imgs.shape[0]):
            colormaps[i] = cv2.resize((cv2.applyColorMap(imgs[i], cv2.COLORMAP_HOT)), (size[1], size[0]))

        # transfer to tensor than to gpu
        # firstly the channel BGR->RGB
        colormaps = colormaps[..., [2, 1, 0]]

        # than to tensor, to gpu
        colormaps = torch.from_numpy(colormaps).cuda()

        return colormaps

    def crop_result(self, result, input_h, input_w, times=32):
        crop_h = crop_w = 0

        if input_h % times != 0:
            crop_h = times - (input_h % times)

        if input_w % times != 0:
            crop_w = times - (input_w % times)

        if crop_h != 0:
            result = result[..., :-crop_h, :]

        if crop_w != 0:
            result = result[..., :-crop_w]
        return result

    def pad_input(self, input, times=32):
        input_h, input_w = input.shape[2:]
        pad_h = pad_w = 0

        if input_h % times != 0:
            pad_h = times - (input_h % times)

        if input_w % times != 0:
            pad_w = times - (input_w % times)

        # print(pad_h, pad_w)

        input = torch.nn.functional.pad(input, (0, pad_w, 0, pad_h), mode='reflect')

        return input

    def minmax_depth(self, depth, blur=True):
        n, c, h, w = depth.shape
        # depth = F.avg_pool2d(depth,kernel_size=5)

        if blur:
            depth = F.pad(depth, [4, 4, 4, 4], 'reflect')
            depth = kornia.filters.median_blur(depth, (9, 9))
            depth = depth[:, :, 3:h - 3, 3:w - 3]

        D_max = torch.max(depth.reshape(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        D_min = torch.min(depth.reshape(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)

        depth = (depth - D_min) / (D_max - D_min + 0.01)

        return depth

