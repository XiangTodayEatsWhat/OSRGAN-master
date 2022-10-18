import os
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index1):
        path1 = self.imgs[index1][0]
        img1 = self.loader(path1)
        if self.transform is not None:
            img1 = self.transform(img1)

        return img1


def return_data(args):
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.img_size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    root = os.path.join(dset_dir, 'CelebA')
    train_kwargs = {'root': root, 'transform': transform}
    dset = CustomImageFolder
    train_data = dset(**train_kwargs)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    data_loader = train_loader
    return data_loader
