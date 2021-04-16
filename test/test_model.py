import unittest
import torch
from models.model import Generator, Discriminator


class TestModels(unittest.TestCase):

    def setUp(self):
        self.N = 8
        self.in_channels = 3
        self.H = 128
        self.W = 128
        self.noise_dim = 128
        self.x = torch.randn((self.N, self.in_channels, self.H, self.W))
        self.z = torch.randn((self.N, self.noise_dim, 1, 1))

    def test_shape_d(self):
        disc = Discriminator(self.in_channels, 8)
        self.assertEqual(disc(self.x).shape, (self.N, 1, 1, 1))
        print(f"{disc(self.x).shape} == {(self.N, 1, 1, 1)}")

    def test_shape_g(self):
        gen = Generator(self.noise_dim, self.in_channels, 8)
        self.assertEqual(gen(self.z).shape, (self.N, self.in_channels, self.H, self.W))
        print(f"{gen(self.z).shape} == {(self.N, self.in_channels, self.H, self.W)}")


if __name__ == '__main__':
    unittest.main()