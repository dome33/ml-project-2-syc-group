import os
import sys
sys.path.append(os.path.abspath(".."))
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath("../../.."))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from urllib.request import urlopen
import torch.optim as optim
from torchsummaryX import summary

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

#from src.models.cnn_bilstm import CNNBILSTM
from src.models.HTR_VT import MaskedAutoencoderViT, create_model
from argparse import ArgumentParser 
import yaml 
import numpy as np 
from types import SimpleNamespace
from src.utils import split_data_provider 

# LOAD CONFIGS 
parser = ArgumentParser() 
parser.add_argument("--config", type=str, required=True) 
args = parser.parse_args() 
config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
configs = SimpleNamespace(**config) 


# LOAD THE DATASET. 
train_dataset = np.load("data/trainset.npy") 
train_dataset = [(name, label) for name, label in train_dataset] 
val_dataset = np.load("data/valset.npy") 
val_dataset = [(name, label) for name, label in val_dataset]

# get the vocabulary 
vocab = [c for _,label in train_dataset for c in label] 
vocab = list(set(vocab)) 
vocab.sort() 
max_len = max([len(label) for _,label in train_dataset])


# Save vocab and maximum text length to configs
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len


# Create a data provider for the dataset
train_dataProvider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

val_dataProvider = DataProvider(
    dataset = val_dataset,
    skip_validation=train_dataProvider._skip_validation,
    batch_size=train_dataProvider._batch_size,
    data_preprocessors=train_dataProvider._data_preprocessors,
    transformers=train_dataProvider._transformers,
    use_cache=train_dataProvider._use_cache,
)


# Augment training data with random brightness, rotation and erode/dilate
train_dataProvider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    RandomRotate(angle=10), 
]


# Create our model 
# NOTE to test another model architecture. add a filed "model" to the configs file 
# and add if-else logic here to create the model specified in the configs file. 
network = create_model(nb_cls=80, img_size=[128,32])
loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(network.parameters(), lr=configs.learning_rate)


# put on cuda device if available
network = network.to(configs.device)

# create callbacks
# used to track important metrics. 
earlyStopping = EarlyStopping(monitor="val_CER", patience=20, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
tb_callback = TensorBoard(configs.model_path + "/logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.height, configs.width, 3), 
    verbose=1,
    metadata={"vocab": configs.vocab}
)


# create model object that will handle training and testing of the network
# NOTE : FOR REASONS I DONT KNOW (YET), DATA IS PASSED TO THE MODEL WITH SHAPE 
# (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS) 
# adapt your model accordingly (look into NOTE of src/models/cnn_bilstm.py) 
model = Model(network, optimizer, loss, metrics=[CERMetric(configs.vocab), WERMetric(configs.vocab)])
model.fit(
    train_dataProvider, 
    val_dataProvider, 
    epochs=1000, 
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr, model2onnx]
    )

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))