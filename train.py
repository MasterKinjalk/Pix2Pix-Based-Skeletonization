import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples, calculate_mccorr
import torch.nn as nn
import torch.optim as optim
import config
from dataset import SkeletonDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

torch.backends.cudnn.benchmark = True


class WeightedFocalLoss(nn.Module):
    def __init__(self, w_pos=50, w_neg=0.75, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.w_pos = w_pos
        self.w_neg = w_neg
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        targets = targets.float()

        loss_pos = (
            -self.w_pos
            * targets
            * (1 - inputs) ** self.gamma
            * torch.log(inputs + 1e-8)
        )
        loss_neg = (
            -self.w_neg
            * (1 - targets)
            * inputs**self.gamma
            * torch.log(1 - inputs + 1e-8)
        )

        return (loss_pos + loss_neg).mean()


def train_fn(
    disc,
    gen,
    loader,
    opt_disc,
    opt_gen,
    weighted_focal_loss,
    bce,
    g_scaler,
    d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            G_focal_loss = weighted_focal_loss(y_fake, y)
            G_loss = G_fake_loss + G_focal_loss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                G_loss=G_loss.item(),
            )


def main():
    evaluation_folder = os.path.join(os.getcwd(), "evaluation")
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    weighted_focal_loss = WeightedFocalLoss(w_pos=50, w_neg=0.75, gamma=2)

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,
            disc,
            opt_disc,
            config.LEARNING_RATE,
        )

    train_dataset = SkeletonDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = SkeletonDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    best_mccorr = 0
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc,
            gen,
            train_loader,
            opt_disc,
            opt_gen,
            weighted_focal_loss,
            BCE,
            g_scaler,
            d_scaler,
        )

        if config.SAVE_MODEL and epoch % 5 == 0:
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            gen_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_GEN)
            disc_path = os.path.join(config.CHECKPOINT_DIR, config.CHECKPOINT_DISC)
            save_checkpoint(gen, opt_gen, filename=gen_path)
            save_checkpoint(disc, opt_disc, filename=disc_path)

        save_some_examples(gen, val_loader, epoch, folder="evaluation")

        # Validation using M-CCORR metric every 10 epochs
        if epoch % 10 == 0:
            current_mccorr = calculate_mccorr(gen, val_loader)
            print(f"Epoch {epoch}, M-CCORR: {current_mccorr:.4f}")

            if current_mccorr > best_mccorr:
                best_mccorr = current_mccorr
                print(f"New best M-CCORR: {best_mccorr:.4f}")
                best_gen_path = os.path.join(
                    config.CHECKPOINT_DIR, "best_generator.pth.tar"
                )
                best_disc_path = os.path.join(
                    config.CHECKPOINT_DIR, "best_discriminator.pth.tar"
                )
                save_checkpoint(gen, opt_gen, filename=best_gen_path)
                save_checkpoint(disc, opt_disc, filename=best_disc_path)


if __name__ == "__main__":
    main()
