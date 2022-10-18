import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class dSpritesDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform

    def __getitem__(self, index1):
        img1 = self.data_tensor[index1]
        if self.transform is not None:
            img1 = self.transform(img1)

        return img1

    def __len__(self):
        return self.data_tensor.size(0)


def return_data(args):
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.img_size
    root = os.path.join(dset_dir, 'dSprites')
    imgs, latent_values, metadata = load_dSprites(root, image_size)
    dsprites_dataset = dSpritesDataset(torch.from_numpy(imgs))
    train_loader = DataLoader(
        dsprites_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    return imgs, train_loader, latent_values, metadata


# load dSprites dataset
def load_dSprites(path, image_size):
    dataset_zip = np.load(
        os.path.join(
            path, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
        allow_pickle=True, encoding='latin1')
    imgs = dataset_zip['imgs']
    latent_values = dataset_zip['latents_values'][:, 1:]
    latent_values = (latent_values - np.min(latent_values, axis=0)) / \
                    (np.max(latent_values, axis=0) - np.min(latent_values, axis=0)) * 2 - 1
    metadata = dataset_zip['metadata'][()]

    imgs = imgs.reshape(737280, 1, image_size, image_size).astype(np.float32)  # 0 ~ 1
    return imgs, latent_values, metadata
