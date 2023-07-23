Depth-aware Unpaired Video Dehaizng
===============================================
This is the PyTorch implementation of the paper 'Depth-aware Unpaired Video Dehaizng'.

Prerequisites
---------------------------------
* Python 3.7
* Pytorch
* NVIDIA GPU + CUDA cuDNN

The detailed prerequiesites are in `environment.yml`

Datasets
---------------------------------
### 1.Data for testing
After downloading the dataset, please use scripts/flist.py to generate the file lists. For example, to generate the file list on the revide testset, you should run:

```
python scripts/flist.py --path path_to_REVIDE_hazy_path --output ./datasets/revide_test.flist
```

Please notice that the ground truth images of SOTS-indoor have additional white border, you can crop it first.


Getting Started
--------------------------------------
To use the pre-trained models, download it from the following link then copy it to the corresponding checkpoints folder. For instance, if you want to test the model on nyu/real-world hazy frames, download the pretrained model for nyu/real-world and put it under  `./checkpoints/test_real`

[Pretrained model on REVIDE](https://drive.google.com/file/d/1E1E_4oK7e1YTYOd3WzQ9wI7PWAVp5M1O/view?usp=drive_link) | [Pretrained model on NYU-depth/Real-world](https://drive.google.com/file/d/1gF6PBdCHSSq6jkkeLGB5Ag0oMGOTJRyN/view?usp=drive_link)



### 2. Testing
1)Prepare the testing datasets following the operations in the Datasets part.
2)Put the trained weight in the checkpoint folder 
3)Add a config file 'config.yml' in your checkpoints folder. We have provided example checkpoints folder and config files in `./checkpoints/`, 
4)Test the model, for example:
```
python test.py --model 1 --checkpoints ./checkpoints/test_revide
```
For quick testing, you can download the checkpoint on real-world frames and put it to the corresponding folder `./checkpoints/test_real` and run test on our example frames directly using

```
python test.py --model 1 --checkpoints ./checkpoints/test_real
```
