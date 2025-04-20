import os
from PIL import Image
from torch.utils.data import Dataset


class IAMWordsDataset(Dataset):
    def __init__(self, words_txt_path, root_dir, transform=None):
        """
        Args:
            words_txt_path (str): Path to 'words.txt' file.
            root_dir (str): Root directory of the word images (i.e., 'words/').
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        with open(words_txt_path, 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue  # Skip comments

                parts = line.strip().split()
                if len(parts) >= 9 and parts[1] == "ok":
                    file_id = parts[0]
                    label = ' '.join(parts[8:])
                    file_path = os.path.join(
                        root_dir,
                        file_id.split("-")[0],
                        file_id.split("-")[0] + "-" + file_id.split("-")[1],
                        file_id + ".png"
                    )

                    abs_path = os.path.abspath(file_path)

                    if os.path.isfile(abs_path):
                        self.samples.append((abs_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # Grayscale

        if self.transform:
            image = self.transform(image)

        return image, label

    def getallsamples(self):
        return self.samples
