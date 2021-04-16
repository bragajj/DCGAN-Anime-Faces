import time
import torch
import torch.nn as nn
import argparse
import wandb
import torch.optim as optim
from config import cfg
from data.datasets import AnimeFacesDataset
from torch.utils.data import DataLoader
from models.model import Generator, Discriminator, init_weights
from metriclogger import MetricLogger
from utils import checkpoint, load_checkpoint, set_seed, get_random_noise


def parse_args():
    parser = argparse.ArgumentParser(description='Anime-DCGAN')
    parser.add_argument('--data_path', dest='data_path', help='path to dataset folder', default=None, type=str)
    parser.add_argument('--checkpoint_path', dest='checkpoint_path', help='path to checkpoint.pth.tar', default=None, type=str)
    parser.add_argument('--out_path', dest='out_path', help='path to output folder', default=None, type=str)
    return parser.parse_args()


def epoch_time(f):
    """Calculate time of each epoch"""
    def timed(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        print("epoch time: %2.1f min" % ((te-ts)/60))
        return result
    return timed


@epoch_time
def train_one_epoch(epoch, dataloader, gen, disc, criterion, opt_gen, opt_disc,
                    fixed_noise, device, metric_logger, num_samples, freq=100):
    """
    Train one epoch
    :param epoch: ``int`` current epoch
    :param dataloader: object of dataloader
    :param gen: Generator model
    :param disc: Discriminator model
    :param criterion: Loss function (for this case: binary cross entropy)
    :param opt_gen: Optimizer for generator
    :param opt_disc: Optimizer for discriminator
    :param fixed_noise: ``tensor[[cfg.BATCH_SIZE, latent_space_dimension, 1, 1]]``
    fixed noise (latent space) for image metrics
    :param device: cuda device or cpu
    :param metric_logger: object of MetricLogger
    :param num_samples: ``int`` well retrievable sqrt() (for example: 4, 16, 64) for good result,
    number of samples for grid image metric
    :param freq: ``int``, must be < len(dataloader)`` freq for display results
    """
    for batch_idx, img in enumerate(dataloader):
        real = img.to(device)
        noise = get_random_noise(cfg.BATCH_SIZE, cfg.Z_DIMENSION, device)
        fake = gen(noise)

        # Train discriminator: We maximize log(D(x)) + log(1 - D(G(z))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train generator: We minimize log(1 - D(G(z))). This is the same as maximize log(D(G(z))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # logs metrics
        if batch_idx % freq == 0:
            with torch.no_grad():
                metric_logger.log(loss_disc, loss_gen, disc_real, disc_fake)
                fake = gen(fixed_noise)
                metric_logger.log_image(fake, num_samples, epoch,
                                        batch_idx, len(dataloader))
                metric_logger.display_status(epoch, cfg.NUM_EPOCHS, batch_idx,
                                             len(dataloader), loss_disc, loss_gen,
                                             disc_real, disc_fake)


if __name__ == '__main__':
    args = parse_args()
    assert args.data_path, 'dataset not specified'

    if args.out_path:
        cfg.OUT_DIR = args.out_path
        cfg.SAVE_CHECKPOINT_PATH = args.out_path
    # set random seed
    set_seed(28)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=> Called with args {args.__dict__}")
    print(f"=> Config params {cfg.__dict__}")
    print(f"=> Run on device {device}")
    # define dataset and dataloader
    dataset = AnimeFacesDataset(args.data_path)
    cfg.DATASET_SIZE = len(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
    # define models
    gen = Generator(cfg.Z_DIMENSION, cfg.CHANNELS_IMG, cfg.FEATURES_GEN).to(device)
    disc = Discriminator(cfg.CHANNELS_IMG, cfg.FEATURES_DISC).to(device)
    # define optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=cfg.LEARNING_RATE, betas=(0.5, 0.999))

    if args.checkpoint_path:
        cp = torch.load(args.checkpoint_path)
        start_epoch, end_epoch, fixed_noise = load_checkpoint(cp, gen, disc, opt_gen, opt_disc)
        cfg.NUM_EPOCHS = end_epoch
    else:
        print("=> Init default weights of models and fixed noise")
        init_weights(gen)
        init_weights(disc)
        start_epoch = 1
        end_epoch = cfg.NUM_EPOCHS
        fixed_noise = get_random_noise(cfg.BATCH_SIZE, cfg.Z_DIMENSION, device)

    # define binary cross entropy.
    # Can be changed to BCEWithLogitsLoss or custom logits loss, then must to replace sigmoid layer from
    # generator to flatten.
    criterion = nn.BCELoss()

    metric_logger = MetricLogger(cfg.PROJECT_VERSION_NAME)

    # gradients metric
    wandb.watch(gen)
    wandb.watch(disc)
    # model mode
    gen.train()
    disc.train()

    start_time = time.time()
    for epoch in range(start_epoch, end_epoch + 1):
        train_one_epoch(epoch, dataloader, gen, disc, criterion, opt_gen, opt_disc,
                        fixed_noise, device, metric_logger, num_samples=cfg.NUM_SAMPLES, freq=cfg.FREQ)
        if epoch == cfg.NUM_EPOCHS + 1:
            checkpoint(epoch, end_epoch, gen, disc, opt_gen, opt_disc, fixed_noise)
        elif epoch % cfg.SAVE_EACH_EPOCH == 0:
            checkpoint(epoch, end_epoch, gen, disc, opt_gen, opt_disc, fixed_noise)

    total_time = time.time() - start_time
    print(f"=> Training time:{total_time}")
