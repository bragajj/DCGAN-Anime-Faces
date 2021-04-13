import torch
import argparse
from tqdm import tqdm
from data.datasets import AnimeFacesDataset
from torch.utils.data import DataLoader


def get_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(dataloader, total=len(dataloader)):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std


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
