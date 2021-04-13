import torch
import os
import argparse
from tqdm import tqdm
from data.datasets import AnimeFacesDataset
from torch.utils.data import DataLoader
from config import cfg


def get_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(dataloader, total=len(dataloader)):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std


def checkpoint(epoch, gen, disc, opt_gen, opt_disc):
    print("=> Saving checkpoint")
    torch.save({
        'gen': gen.state_dict(),
        'disc': disc.state_dict(),
        'opt_gen': opt_gen.state_dict(),
        'opt_disc': opt_disc.state_dict(),
    }, f"{cfg.SAVE_CHECKPOINT_PATH}epoch_{epoch}.pth.tar")
    print(f"=> Checkpoint save to {cfg.SAVE_CHECKPOINT_PATH}")


def load_checkpoint(checkpoint, gen, disc, opt_gen, opt_disc):
    print("=> Load checkpoint...")
    gen.load_state_dict(checkpoint['gen'])
    disc.load_state_dict(checkpoint['disc'])
    opt_gen.load_state_dict(checkpoint['opt_gen'])
    opt_disc.load_state_dict(checkpoint['opt_disc'])
    print("=> Checkpoint loaded")


def load_gen(gen, filename):
    print("=> Load generator...")
    gen_path = os.path.join(os.getcwd(), filename)
    gen.load_state_dict(torch.load(gen_path))
    print(f"=> Generator model loaded from {gen_path}")


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
