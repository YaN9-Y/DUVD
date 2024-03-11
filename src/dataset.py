import os
import glob
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from PIL import Image, ImageFile
import torchvision.transforms as transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, crop_size,  hazy_flist, clean_flist=None, augment=True, split='unpair'):
        super(Dataset, self).__init__()
        self.augment = augment
        self.config = config
        assert split in ['unpair', 'pair_test', 'hazy', 'clean', 'hazy_various']

        self.split = split


        self.clean_data = self.load_flist(clean_flist)
        self.noisy_data = self.load_flist(hazy_flist)

        self.input_size = crop_size

        self.transforms = transforms.Compose(([
                                                  transforms.RandomCrop((self.input_size, self.input_size)),
                                              ] if crop_size else [])
                                             + ([transforms.RandomHorizontalFlip()]
                                                if self.augment else [])
                                             + [transforms.ToTensor()]
                                             )
        print(self.clean_data)

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        if self.split in ['clean','depth']:
            name = self.clean_data[index]
        else:
            name = self.noisy_data[index]
        return os.path.basename(name)

    def load_item(self, index):
        # load image
            if self.split in ['hazy','hazy_various']:
                img_noisy = Image.open(self.noisy_data[index])
                img_noisy = self.convert_to_rgb(img_noisy)
                img_noisy = TF.to_tensor(img_noisy)
                return img_noisy

            elif self.split in ['clean','depth']:
                img_clean = Image.open(self.clean_data[index])
                img_clean = self.convert_to_rgb(img_clean)
                img_clean = TF.to_tensor(img_clean)

                return img_clean


            elif self.split in ['unpair']:
                while (True):
                    clean_index = int(np.random.random() * len(self.clean_data))
                    img_clean = Image.open((self.clean_data[clean_index]))
                    if np.array(img_clean).shape is None:
                        print(self.clean_data[clean_index])
                    if min(np.array(img_clean).shape[0:2]) > self.config.CROP_SIZE and clean_index != index:
                        break

                while(True):
                    noisy_index = index
                    img_noisy = Image.open((self.noisy_data[noisy_index]))
                    if np.array(img_noisy).shape is None:
                        print(self.noisy_data[noisy_index])
                    if min(np.array(img_noisy).shape[0:2]) > self.config.CROP_SIZE:
                        break

                img_noisy = self.convert_to_rgb(img_noisy)
                img_clean = self.convert_to_rgb(img_clean)



                img_clean = self.transforms(img_clean)
                img_noisy = self.transforms(img_noisy)

                return img_clean, img_noisy, img_clean, img_clean

            elif self.split in ['pair_train', 'pair_test']:

                img_noisy = Image.open(self.noisy_data[index])
                img_clean = Image.open(self.clean_data[index])

                img_noisy = self.convert_to_rgb(img_noisy)
                img_clean = self.convert_to_rgb(img_clean)

                if img_clean.size != img_noisy.size:
                    img_clean = TF.center_crop(img_clean, img_noisy.size[::-1])

                img_clean = TF.to_tensor(img_clean)
                img_noisy = TF.to_tensor(img_noisy)


                return img_clean, img_noisy #, gt_grad



    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))+list(glob.glob(flist + '/*.jpeg'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')

        return []


    def load_image_to_memory(self, flist):
        filelist = np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
        images_list = []
        for i in range(len(filelist)):
            images_list.append(np.array(Image.open(filelist[i])))
            if i%100 == 0:
                print('loading data: %d / %d', i+1, len(filelist))
        return images_list

    def create_iterator(self, batch_size, shuffle=False):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True,
                shuffle=shuffle
            )

            for item in sample_loader:
                yield item


    def convert_to_rgb(self, img):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return img











