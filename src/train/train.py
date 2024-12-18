import os
import torch.optim as optim
from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from src.utils import CERMetricShortCut, CTCLossShortcut, WERMetricShortCut 
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage
from src.augmentations import RandomRotateFillWithMedian 
from src.models.cnn_bilstm import CNNBILSTM

#from src.models.cnn_bilstm import CNNBILSTM
from src.models.HTR_VT import MaskedAutoencoderViT, create_model
from argparse import ArgumentParser 
import yaml 
import numpy as np 
from types import SimpleNamespace 
from src.models.htr_net import HTRNet


possible_models = ["cnn_bilstm", "htr_net", "htr_vit"] 


# LOAD CONFIGS 
parser = ArgumentParser() 
parser.add_argument("--config", type=str, required=True) 
args = parser.parse_args() 
config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
configs = SimpleNamespace(**config) 

assert configs.model is not None, "Please specify a model architecture in the configs file"
assert configs.model in possible_models, f"Model {configs.model} not found. Possible models are {possible_models}" 


# LOAD THE DATASET. 
train_dataset = np.load("data/trainset.npy") 
train_dataset = [(name, label) for name, label in train_dataset] 
val_dataset = np.load("data/valset.npy") 
val_dataset = [(name, label) for name, label in val_dataset]


# Get the vocabulary 
vocab = [c for _,label in train_dataset for c in label] 
vocab = list(set(vocab)) 
vocab.sort() 
max_len = max([len(label) for _,label in train_dataset])


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
    RandomRotateFillWithMedian(angle=10), 
]


# Create our model 
model2network = {
    "cnn_bilstm": CNNBILSTM(len(configs.vocab), activation="leaky_relu", dropout=0.3),
    "htr_net": HTRNet(len(configs.vocab)+1, configs.device), 
    "htr_vit": create_model(len(configs.vocab)+1, (configs.height, configs.width)) 
}

network = model2network[configs.model]
loss = CTCLossShortcut(blank=len(configs.vocab)) 
optimizer = optim.Adam(network.parameters(), lr=configs.learning_rate)

# Put on cuda device if available
network = network.to(configs.device)

# Create callbacks used to track important metrics. 
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


# Create model object that will handle training and testing of the network
model = Model(network, optimizer, loss, metrics=[CERMetricShortCut(configs.vocab), WERMetricShortCut(configs.vocab)])
model.fit(
    train_dataProvider, 
    val_dataProvider, 
    epochs=1000, 
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr, model2onnx]
)


# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
val_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))