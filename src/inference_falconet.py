import os

# Get and print the current working directory
current_working_directory = os.getcwd()
print(f"Current working directory: {current_working_directory}")


import numpy as np
import torch
import torch.nn.functional as F

from mltu.torch.dataProvider import DataProvider
from train.preprocessors import ImageReader
from mltu.transformers import ImageResizer,  ImageShowCV2, ImageNormalizer

from mltu.inferenceModel import OnnxInferenceModel

from mltu.annotations.images import CVImage
from models.cnn_owen import HandwritingRecognitionCNN_BiLSTM

from mltu.utils.text_utils import ctc_decoder, get_cer


import pandas as pd
from tqdm import tqdm

class ImageToWordTorchModel:
    
    def __init__(self, model_path: str, vocab: str):
        """
        Args:
            model_path (str): Path to the PyTorch model file (.pt or .pth).
            vocab (List[str]): Vocabulary used for decoding.
            input_shape (tuple): Model input image shape as (height, width).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab = vocab

        # use these params from training
        num_classes = 30
        hidden_size = 64

       # num_classes = 81
       # hidden_size = 256

        # load weights
        self.model = HandwritingRecognitionCNN_BiLSTM(num_classes=num_classes, hidden_size=hidden_size)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def prepare_image(self, image_np: np.ndarray) -> torch.Tensor:
        """
        Convert a grayscale NumPy image (H, W) to a PyTorch tensor (1, 1, H, W).
        """
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).float()
        return image_tensor
    
    def predict(self, image) -> str:
        """
        Predict text from a single preprocessed grayscale image.
        
        Args:
            image (np.ndarray): Preprocessed grayscale image of shape (H, W).
        
        Returns:
            str: Decoded text.
        """
        if self.vocab is None:
            raise ValueError("Vocab must be set using `set_vocab()` before calling predict.")

        input = self.prepare_image(image)

        with torch.no_grad():
            output = self.model(input)  # output shape: (B, T, C), already log-softmaxed
            output_np = output.cpu().numpy()

        out = ctc_decoder(output_np, self.vocab)

        return out[0]
    
class ImageToWordModel(OnnxInferenceModel):
    
    def __init__(self, model_path: str, vocab: str):
        """
        Initialize the ImageToWordModel by calling the parent class's initializer.
        The model expects metadata and input-output specifications to handle image-to-text predictions.
        """
        super().__init__(model_path=model_path)

        self.vocab = vocab

    
    def prepare_image(self, image_np: np.ndarray) -> torch.Tensor:
        """
        Convert a grayscale NumPy image (H, W) to a PyTorch tensor (1, 1, H, W).
        """
        image_tensor = np.expand_dims(image, axis=0).astype(np.float32)
        return image_tensor


    def predict(self, image: np.ndarray):
        """
        Perform text prediction on a given image using the ONNX model.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            str: Decoded text prediction.
        """

        input = self.prepare_image(image)

        k = self.input_names[0]
        # Run the model and get the predictions
        preds = self.model.run(self.output_names, {k: input})[0]

        # Decode the predictions
        text = ctc_decoder(preds, self.vocab)[0]

        return text


if __name__ == "__main__":

    configs = {
        "batch_size": 1,
        "width": 256,
        "height": 64,
        "vocab": ""
    }

    # Load vocab and model path
    # LOAD THE DATASET. 
    
    train_dataset = np.load("data/trainset.npy")
    train_dataset = [(name, label) for name, label in train_dataset]
    val_dataset = np.load("data/testset.npy")
    val_dataset = [(name, label) for name, label in val_dataset]
    
    # Get the vocabulary
    vocab = [c for _, label in train_dataset for c in label]
    vocab = list(set(vocab))
    vocab.sort()
    #max_len = max([len(label) for _, label in train_dataset])
    max_len = 12

    # Save vocab and maximum text length to configs
    vocab = "".join(sorted(vocab))
    
    model_path = "results/falconet/falconet_model.onnx"
    
    #model = ImageToWordTorchModel(model_path=model_path, vocab=vocab)
    model = ImageToWordModel(model_path=model_path, vocab=vocab)

    val_dataProvider = DataProvider(
        dataset=val_dataset,
        skip_validation=True,
        batch_size=configs["batch_size"],
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(configs["width"], configs["height"], keep_aspect_ratio=True, padding_color=(255, 255, 255)),
            ImageNormalizer(transpose_axis=False)
        ],
        use_cache=False,
    )

    accum_cer = []

    for batch in tqdm(val_dataProvider):
        image, label = batch[0][0], batch[1][0]  # Get image and text label

        prediction_text = model.predict(image)
        cer = get_cer(prediction_text, label)

        print(f"Prediction: {prediction_text}, Label: {label}, CER: {cer}")
        accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")
