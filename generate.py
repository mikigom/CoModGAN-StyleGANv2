import argparse
from tqdm import tqdm

import torch
from torch.utils import data
from torchvision import transforms, utils

from train import data_sampler, sample_data
from dataset import MultiResolutionDataset
from model import Generator
from mask_generator import RandomMaskGenerator


def generate(args, g_ema, device, mean_latent, loader, random_mask_generator):
    loader = sample_data(loader)

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            real_img = next(loader)
            real_img = real_img.to(device)
            sample_mask = random_mask_generator()
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema(
                torch.cat((real_img, sample_mask), dim=1), [sample_z],
                truncation=args.truncation, truncation_latent=mean_latent
            )

            masked_image = real_img * (1 - sample_mask) + torch.ones_like(real_img) * sample_mask
            utils.save_image(
                torch.cat((masked_image, sample), dim=0),
                f"sample/test_{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, 1024, args.n_mlp, bundle_channels=3, label_channels=1,
        channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)

    g_ema.load_state_dict(checkpoint["g_ema"])

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.sample,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True,
    )

    random_mask_generator = RandomMaskGenerator(args.sample, args.size, device)

    generate(args, g_ema, device, mean_latent, loader, random_mask_generator)
