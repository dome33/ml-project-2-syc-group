import os

import cv2
import torch.optim as optim
from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from torch.utils.data import DataLoader

import torch

from mltu.torch.metrics import CERMetric, WERMetric

#from src.augmentations import RandomRotateFillWithMedian

from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau

from src.train.preprocessors import ImageReader
from mltu.transformers import ImageResizer,  ImageShowCV2, ImageNormalizer

from src.train.transformers import LabelIndexer, LabelPadding

from src.train.augmentors import RandomBrightness, RandomRotate
from mltu.annotations.images import CVImage

from src.models.cnn_owen import HandwritingRecognitionCNN_BiLSTM


from argparse import ArgumentParser
import yaml
import numpy as np
from types import SimpleNamespace


from src.train.iamwordsdataset import IAMWordsDataset

from src.train.iamtransform import transform

possible_models = ["cnn_owen"]

# LOAD CONFIGS
parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
args = parser.parse_args()
config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
configs = SimpleNamespace(**config)

assert configs.model is not None, "Please specify a model architecture in the configs file"
assert configs.model in possible_models, f"Model {configs.model} not found. Possible models are {possible_models}"

#dataset = IAMWordsDataset(
#    words_txt_path="data/iam_dataset/ascii/words.txt",
#    root_dir="data/iam_dataset/words",
#    transform=transform
#)

#loader = DataLoader(dataset, batch_size=32, shuffle=True)

#for images, labels in loader:
#    print(images.shape, labels[0])
#    break


# LOAD THE DATASET. 
train_dataset = np.load("data/trainset.npy")
train_dataset = [(name, label) for name, label in train_dataset]
val_dataset = np.load("data/valset.npy")
val_dataset = [(name, label) for name, label in val_dataset]

# Get the vocabulary
vocab = [c for _, label in train_dataset for c in label]
vocab = list(set(vocab))
vocab.sort()
#max_len = max([len(label) for _, label in train_dataset])
max_len = 31

# Save vocab and maximum text length to configs
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
print(f"Len of train dataset: {len(train_dataset)}")

# Create a data provider for the dataset
train_dataProvider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=True, padding_color=(255, 255, 255)),
        ImageNormalizer(transpose_axis=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
    ],
    use_cache=True,
)

val_dataProvider = DataProvider(
    dataset=val_dataset,
    skip_validation=train_dataProvider._skip_validation,
    batch_size=train_dataProvider._batch_size,
    data_preprocessors=train_dataProvider._data_preprocessors,
    transformers=train_dataProvider._transformers,
    use_cache=train_dataProvider._use_cache,
)



#for step, (data, target) in enumerate(val_dataProvider, start=1):
#    targets.append(target)


#  RandomBrightness(),
# RandomSharpen()
# Augment training data with random brightness, rotation and erode/dilate
train_dataProvider.augmentors = [
    RandomBrightness(random_chance=0.3),
    RandomRotate(random_chance=0.3, angle=10)
]

num_classes = 81
hidden_size = 256

# Create our model
model2network = {
    "cnn_owen": HandwritingRecognitionCNN_BiLSTM(num_classes=num_classes, hidden_size=hidden_size)
}

blank = len(configs.vocab)

network = model2network[configs.model]
loss = CTCLoss(blank=blank, zero_infinity=True)


optimizer = optim.Adam(network.parameters(), lr=configs.learning_rate)

# Put on cuda device if available
network = network.to(configs.device)

network.load_state_dict(torch.load(configs.model_path + "/model.pt"))

# Create callbacks used to track important metrics. 
earlyStopping = EarlyStopping(monitor="val_CER", patience=20, mode="min", verbose=1)

modelCheckpoint = ModelCheckpoint(
    configs.model_path + "/model.pt", monitor="val_CER",
    mode="min",
    save_best_only=True,
    verbose=1)

tb_callback = TensorBoard(configs.model_path + "/logs")

reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)

model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(configs.batch_size, 1, configs.height, configs.width),
    verbose=1,
    metadata={"vocab": configs.vocab}
)

# Create model object that will handle training and testing of the network
model = Model(network, optimizer, loss, metrics=[CERMetric(configs.vocab), WERMetric(configs.vocab)])

# Output shape should be (sequence_length, batch_size, num_classes)

model.fit(
    train_dataProvider,
    val_dataProvider,
    epochs=9,
    initial_epoch=1,
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr, model2onnx]
)

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))
