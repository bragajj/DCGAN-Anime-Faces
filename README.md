## DCGAN-Anime-Faces
PyTorch implementation of DCGAN introduced in the paper: [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434), Alec Radford, Luke Metz, Soumith Chintala.

## Hyperparameters
All hyperparameters of this implementation specified in config file config.py

## Dataset
Original Dataset: https://www.kaggle.com/soumikrakshit/anime-faces
![dataset](https://raw.githubusercontent.com/ErrorInever/DCGAN-Anime-Faces/master/data/image_demonstration/Figure_1.png)

## Results
Training time one hour (GPU Tesla P100-PCIE-16GB)
![loss](https://github.com/ErrorInever/DCGAN-Anime-Faces/blob/master/data/image_demonstration/W%26B%20Chart%204_22_2021%2C%2010_27_45%20PM.png)


![batch](https://github.com/ErrorInever/DCGAN-Anime-Faces/blob/master/data/image_demonstration/res.png)


Interpolation

![int](https://github.com/ErrorInever/DCGAN-Anime-Faces/blob/master/data/image_demonstration/__results___27_1.png)


Latent space interpolation (I specially selected good samples for demonstration)

![z_int](https://github.com/ErrorInever/DCGAN-Anime-Faces/blob/master/data/image_demonstration/int_z_dim.gif)


## ARGS and runs
    optional arguments:
      --data_path            path to dataset folder
      --seed                 seed value, default=7889
      --checkpoint_path      path to checkpoint.pth.tar
      --out_path             path to output folder
      --resume_id            wandb id of project for resume metric
      --device               use device, can be - cpu, cuda, tpu, if not specified: use gpu if available

      Other paths and other parameters you can set up in config.py
   > for example: python3 train.py --data_path 'anime_dataset'
    
   
## Inference
    optional arguments:
      --path_ckpt            Path to checkpoint of model
      --num_samples          Number of samples
      --steps                Number of step interpolation
      --device               cpu or gpu
      --out_path             Path to output folder, default=save to project folder
      --gif                  reate gif
      --grid                 Draw grid of images
      --z_size               The size of latent space, default=128
      --img_size             Size of output image
      --resize               if you want to resize images

   > You can use pretrained weights,
   > for example: python3 'DCGAN-Anime-Faces/inference.py' --path_ckpt 'data/weights/DCGAN_epoch_50.pth.tar' --num_samples 15 --steps 20 --gif True --resize 128
