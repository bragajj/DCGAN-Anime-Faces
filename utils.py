import torch
import os
import random
import numpy as np
import argparse
from tqdm import tqdm
from data.datasets import AnimeFacesDataset
from torch.utils.data import DataLoader
from config import cfg


# not working if generator has tanh() output, because values go outside [-1, 1], therefore generator limited
def get_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data in tqdm(dataloader, total=len(dataloader)):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std


def get_random_noise(size, dim, device):
    """
     Get random noise from normal distribution
    :param size: ``int``, number of samples (batch)
    :param dim: ``int``, dimension
    :param device: cuda or cpu device
    :return: Tensor([size, dim, 1, 1])
    """
    return torch.randn(size, dim, 1, 1).to(device)


def checkpoint(epoch, end_epoch, gen, disc, opt_gen, opt_disc, fixed_noise):
    print("=> Saving checkpoint")
    torch.save({
        'gen': gen.state_dict(),
        'disc': disc.state_dict(),
        'opt_gen': opt_gen.state_dict(),
        'opt_disc': opt_disc.state_dict(),
        'start_epoch': epoch,
        'end_epoch': end_epoch,
        'fixed_noise': fixed_noise
    }, f"{cfg.SAVE_CHECKPOINT_PATH}/DCGAN_epoch_{epoch}.pth.tar")
    print(f"=> Checkpoint save to {cfg.SAVE_CHECKPOINT_PATH}")


def load_checkpoint(checkpoint, gen, disc, opt_gen, opt_disc):
    print("=> Load checkpoint...")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])
    opt_gen.load_state_dict(checkpoint['opt_gen'])
    opt_disc.load_state_dict(checkpoint['opt_disc'])
    print("=> Checkpoint loaded")
    return checkpoint['start_epoch'], checkpoint['end_epoch'], checkpoint['fixed_noise']


def load_gen(gen, filename):
    print("=> Load generator...")
    gen_path = os.path.join(os.getcwd(), filename)
    gen.load_state_dict(torch.load(gen_path))
    print(f"=> Generator model loaded from {gen_path}")


def set_seed(val):
    """
    Freezes random sequences
    :param val: ``int`` random value
    """
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    torch.cuda.manual_seed(val)


def parse_args():
    parser = argparse.ArgumentParser(description='Compute mean and std')
    parser.add_argument('--path', dest='path', help='path to image folder of dataset',
                        default=None, type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert args.path, 'path not specified'
    dataset = AnimeFacesDataset(args.path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    mean, std = get_mean_std(dataloader)
    print(f"mean:{mean}  std:{std}")
