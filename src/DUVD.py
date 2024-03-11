import os
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset
from .models import Model
from .utils import Progbar, create_dir,  imsave
from .metrics import PSNR_RGB
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
        self.log_path = os.path.join(config.PATH, 'logs')

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True

        self.log_file = os.path.join(config.PATH, 'log_' + self.model.name + '.dat')

    def load(self):
        self.model.load()

    def save(self, save_best=False, psnr=None, iteration=None):
        self.model.save(save_best, psnr, iteration)

    def train(self):
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )

        keep_training = True
        model = self.config.MODEL
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset)
        epoch = self.model.epoch
        self.loss_list = []
        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while (keep_training):
            epoch += 1
            print('\n\nTraining epoch: %d' % epoch)

            progbar = Progbar(total, width=20, stateful_metrics=['epoch', 'iter'])
            print('epoch:', epoch)

            index = 0

            for items in train_loader:
                self.model.train()
                clean_images, hazy_images, more_clean_images, more_hazy_images = self.cuda(items[0], items[1], items[2],
                                                                                           items[3])

                if model == 1:
                    outputs, gen_loss, dis_loss, logs = self.model.process(clean_images, hazy_images)
                    psnr = self.psnr(self.postprocess(clean_images), self.postprocess(outputs))
                    logs.append(('psnr_cyc', psnr.item()))
                    iteration = self.model.iteration

                elif model == 2:
                    iteration = self.model.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                           ("epoch", epoch),
                           ("iter", iteration),
                       ] + logs

                index += 1
                progbar.add(len(clean_images), values=logs if self.config.VERBOSE else [x for x in logs])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()

            # update epoch for scheduler
            self.model.epoch = epoch
            self.model.update_scheduler()
        print('\nEnd training....')

    def test(self):
        model = self.config.MODEL
        self.model.eval()
        create_dir(self.results_path)
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        index = 0
        with torch.no_grad():
            for items in test_loader:

                if self.test_dataset.split == 'hazy':
                    name = self.test_dataset.load_name(index)[:-4] + '.png'
                    hazy_images = items.to(self.config.DEVICE)
                    index += 1

                    if model == 1:
                        torch.cuda.empty_cache()
                        h, w = hazy_images.shape[2:4]
                        hazy_input_images = self.pad_input(hazy_images)
                        predicted_results = self.model.forward_h2c(hazy_input_images)
                        predicted_results = self.crop_result(predicted_results, h, w)
                        predicted_results = self.postprocess(predicted_results)[0]
                        path = os.path.join(self.results_path, self.model.name)
                        create_dir(path)
                        save_name = os.path.join(path, name)
                        imsave(predicted_results, save_name)
                        print(save_name)

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

        input = torch.nn.functional.pad(input, (0, pad_w, 0, pad_h), mode='reflect')

        return input


