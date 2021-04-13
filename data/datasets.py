import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from config import cfg


class AnimeFacesDataset(Dataset):
    def __init__(self, img_folder):
        self.img_folder = img_folder
        self.img_names = [n for n in os.listdir(img_folder) if n.endswith(('png', 'jpeg', 'jpg'))]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_folder, self.img_names[idx])).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return len(self.img_names)

    @property
    def transform(self):
        return transforms.Compose([transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.7007, 0.6007, 0.5895],
                                                        std=[0.2938, 0.2973, 0.2702])])
