import torch.nn as nn


class Generator(nn.Module):
    """Full convolution generator"""
    def __init__(self, channels_noise, channels_img, features_gen):
        """
        :param channels_noise: input latent space dimension
        :param channels_img: must be 3 for RGB image
        :param features_gen: deep
        """
        super().__init__()
        self.body = nn.Sequential(
            self._default_block(channels_noise, features_gen * 16, 4, 1, 0),    # 4x4
            self._default_block(features_gen * 16, features_gen * 8, 4, 2, 1),  # 8x8
            self._default_block(features_gen * 8, features_gen * 4, 4, 2, 1),   # 16x16
            self._default_block(features_gen * 4, features_gen * 2, 4, 2, 1),   # 32x32
            nn.ConvTranspose2d(
                features_gen * 2, channels_img, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),  # 64x64
            nn.Tanh()
            # out dimension: [N x channels_img x 64 x 64]
        )

    def _default_block(self, in_channels, out_channels, kernel_size, stride, padding):
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
            self._default_block(features_d, features_d * 2, 4, 2, 1),
            self._default_block(features_d * 2, features_d * 4, 4, 2, 1),
            self._default_block(features_d * 4, features_d * 8, 4, 2, 1),
            nn.Conv2d(features_d * 8, 1, kernel_size=(4, 4), stride=(2, 2), padding=(0, 0)),
            nn.Sigmoid()
        )

    def _default_block(self, in_channels, out_channels, kernel_size, stride, padding):
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
