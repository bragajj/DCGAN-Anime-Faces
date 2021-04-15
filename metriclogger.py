import os
import numpy as np
import wandb
import torchvision
import errno
import torch
from matplotlib import pyplot as plt
from IPython import display
from config import cfg


class MetricLogger:
    """Metric class"""
    def __init__(self, project_name, wab=True, show_accuracy=True):
        self.project_name = project_name
        self.show_acc = show_accuracy
        self.data_subdir = f"{os.path.join(cfg.OUT_DIR, self.project_name)}/imgdata"

        if wab:
            wandb_id = wandb.util.generate_id()
            wandb.init(id=wandb_id, project='DCGAN-Anime-Faces', name=project_name)
            wandb.config.update({
                'init_lr': cfg.LEARNING_RATE,
                'noise_z_size': cfg.Z_DIMENSION,
                'batch_size': cfg.BATCH_SIZE,
                'initialization_weights': 'Normal Distribution',
                'beta 1': 0.5,
                'beta 2': 0.999
            })

    def display_status(self, epoch, num_epochs, batch_idx, num_batches, dis_loss,
                       gen_loss, acc_real=None, acc_fake=None):
        """
        Display training progress
        :param epoch: ``int``, current epoch
        :param num_epochs: ``int``, numbers epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        :param dis_loss: ``torch.autograd.Variable``, discriminator loss
        :param gen_loss: ``torch.autograd.Variable``, generator loss
        :param acc_real: ``torch.autograd.Variable``, discriminator predicted on real data
        :param acc_fake: ``torch.autograd.Variable``, discriminator predicted on fake data
        """
        if isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.item()
        if isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()
        if self.show_acc and isinstance(acc_real, torch.autograd.Variable):
            acc_real = acc_real.float().mean().item()
        if self.show_acc and isinstance(acc_fake, torch.autograd.Variable):
            acc_fake = acc_fake.float().mean().item()

        print('Batch Num: [{}/{}], Epoch: [{}/{}]'.format(batch_idx, num_batches, epoch, num_epochs))
        print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(dis_loss, gen_loss))
        if acc_real and acc_fake:
            print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(acc_real, acc_fake))

    def log(self, epoch, batch_idx, num_batches, dis_loss, gen_loss, acc_real=None, acc_fake=None):
        """
        Logging values
        :param epoch: ``int``, current epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, numbers bathes
        :param dis_loss: ``torch.autograd.Variable``, critical loss
        :param gen_loss: ``torch.autograd.Variable``, generator loss
        :param acc_real: ``torch.autograd.Variable``, D(x) predicted on real data
        :param acc_fake: ``torch.autograd.Variable``, D(G(z)) paramredicted on fake data
        """
        if isinstance(dis_loss, torch.autograd.Variable):
            dis_loss = dis_loss.item()
        if isinstance(gen_loss, torch.autograd.Variable):
            gen_loss = gen_loss.item()
        if self.show_acc and isinstance(acc_real, torch.autograd.Variable):
            acc_real = acc_real.float().mean().item()
        if self.show_acc and isinstance(acc_fake, torch.autograd.Variable):
            acc_fake = acc_fake.float().mean().item()

        wandb.log({'d_loss': dis_loss, 'g_loss': gen_loss, 'D(x)': acc_real, 'D(G(z))': acc_fake})
        MetricLogger._step(epoch, batch_idx, num_batches)

    @staticmethod
    def _step(epoch, batch_idx, num_batches):
        return epoch * num_batches + batch_idx

    def log_image(self, images, num_samples, epoch, batch_idx, num_batches, normalize=True):
        """
        Create image grid and save it
        :param images: ``Torch.Tensor(N,C,H,W)``, tensor of images
        :param num_samples: ``int``, number of samples
        :param epoch: ``int``, current epoch
        :param batch_idx: ``int``, current batch
        :param num_batches: ``int``, number of batches
        :param normalize: if True normalize images
        """
        images = images[:num_samples, ...]
        horizontal_grid = torchvision.utils.make_grid(images, normalize=normalize, scale_each=True)
        nrows = int(np.sqrt(num_samples))
        grid = torchvision.utils.make_grid(images, nrow=nrows, normalize=normalize, scale_each=True)
        step = MetricLogger._step(epoch, batch_idx, num_batches)
        self.save_torch_images(horizontal_grid, grid, step)

    def save_torch_images(self, horizontal_grid, grid, step, plot_horizontal=True, figsize=(16, 16)):
        """
        Display and save image grid
        :param horizontal_grid: ``ndarray``, horizontal grid image
        :param grid: ``ndarray``, grid image
        :param step: ``int``, step
        :param plot_horizontal: if True plot horizontal grid image
        :param figsize: ``tuple``, figure size
        """
        out_dir = self.data_subdir
        fig = plt.figure(figsize=figsize)
        plt.imshow(np.moveaxis(horizontal_grid.detach().cpu().numpy(), 0, -1))
        plt.axis('off')
        if plot_horizontal:
            display.display(plt.gcf())
        MetricLogger._save_images(fig, out_dir, step)
        plt.close()
        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.moveaxis(grid.detach().cpu().numpy(), 0, -1), aspect='auto')
        plt.axis('off')
        MetricLogger._save_images(fig, out_dir, step)
        plt.close()

    @staticmethod
    def _save_images(fig, out_dir, step):
        MetricLogger._make_dir(out_dir)
        fig.savefig('{}/img_{}.png'.format(out_dir, step))

    @staticmethod
    def _make_dir(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
