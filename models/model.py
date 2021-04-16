import torch.nn as nn


class Generator(nn.Module):
    """Full convolution generator"""
    def __init__(self, channels_noise, channels_img, features_gen):
        """
        :param channels_noise: ``int``, input latent space dimension
        :param channels_img: ``int``,  3 for RGB image or 1 for GrayScale
        :param features_gen: ``int``, num features of generator
        """
        super().__init__()
        self.body = nn.Sequential(
            Generator._default_block(channels_noise, features_gen * 16, 4, 1, 0),    # 4x4
            Generator._default_block(features_gen * 16, features_gen * 8, 4, 2, 1),  # 8x8
            Generator._default_block(features_gen * 8, features_gen * 4, 4, 2, 1),   # 16x16
            Generator._default_block(features_gen * 4, features_gen * 2, 4, 2, 1),   # 32x32
            Generator._default_block(features_gen * 2, features_gen, 4, 2, 1),       # 64x64
            nn.ConvTranspose2d(
                features_gen, channels_img, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
            # out dimension: [N x 3 x 128 x 128] with range [-1, 1]
        )

    @staticmethod
    def _default_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.body(x)


class Discriminator(nn.Module):
    """Standard full convolution discriminator"""
    def __init__(self, in_channels, features_d):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, features_d, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            Discriminator._default_block(features_d, features_d * 2, 4, 2, 1),
            Discriminator._default_block(features_d * 2, features_d * 4, 4, 2, 1),
            Discriminator._default_block(features_d * 4, features_d * 8, 4, 2, 1),
            Discriminator._default_block(features_d * 8, features_d * 16, 4, 2, 1),
            nn.Conv2d(features_d * 16, 1, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.Sigmoid(),
        )

    @staticmethod
    def _default_block(in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.body(x)


def init_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
