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


Latent space interpolation

![z_int](https://github.com/ErrorInever/DCGAN-Anime-Faces/blob/master/data/image_demonstration/int_z_dim.gif)


## ARGS and runs
    Args:
        --data_path : path to dataset folder
        --seed: set seed value
        --checkpoint_path: for resume training, path to checkpoint *.pth.tar 
        --out_path: path for output data
        --resume_id: resume id for metric Weights & Biases (optional)
        --device: if you use TPU set it to 'TPU'
        
        for example: train.py --data_path /home/animedataset
        
    Other paths and other parameters you can set up in config.py
