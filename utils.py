import torch
import config
from torchvision.utils import save_image
from multipart_mccorr import calculate_mccorr as mccorr_metric
import numpy as np
import os


def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        try:
            save_image(y_fake, os.path.join(folder, f"y_gen_{epoch}.png"))
            save_image(x * 0.5 + 0.5, os.path.join(folder, f"input_{epoch}.png"))
            if epoch == 1:
                save_image(y * 0.5 + 0.5, os.path.join(folder, f"label_{epoch}.png"))
        except Exception as e:
            print(f"Error saving images: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Target directory: {os.path.abspath(folder)}")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def calculate_mccorr(gen, val_loader):
    total_mccorr = 0
    num_samples = 0

    for x, y in val_loader:
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        with torch.no_grad():
            y_fake = gen(x)

        # Convert tensors to numpy arrays and move channels to the last dimension
        y_true = y.cpu().squeeze().permute(1, 2, 0).numpy()
        y_pred = y_fake.cpu().squeeze().permute(1, 2, 0).numpy()

        # Calculate M-CCORR directly from numpy arrays
        mccorr = mccorr_metric(y_true, y_pred)

        total_mccorr += mccorr
        num_samples += 1

    return total_mccorr / num_samples if num_samples > 0 else 0
