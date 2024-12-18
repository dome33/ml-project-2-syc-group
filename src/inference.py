import cv2
import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

import pandas as pd
from tqdm import tqdm
    
class ImageToWordModel(OnnxInferenceModel):
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the ImageToWordModel by calling the parent class's initializer.
        The model expects metadata and input-output specifications to handle image-to-text predictions.
        """
        super().__init__(*args, **kwargs)


    def predict(self, image: np.ndarray):
        """
        Perform text prediction on a given image using the ONNX model.

        Args:
            image (np.ndarray): Input image array.

        Returns:
            str: Decoded text prediction.
        """

        # Resize the image to the model's input shape
        image = cv2.resize(image, self.input_shapes[0][1:3][::-1])

        # Add batch dimension and convert to float32
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        # Run the model and get the predictions
        preds = self.model.run(self.output_names, {self.input_names[0]: image_pred})[0]

        # Decode the predictions
        text = ctc_decoder(preds, self.metadata["vocab"])[0]

        return text


if __name__ == "__main__":

    model = ImageToWordModel(model_path="results/cnn_bilstm_default/model.onnx") 
    df = pd.read_csv("results/cnn_bilstm_default/val.csv").values.tolist()

    accum_cer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path.replace("\\", "/"))

        prediction_text = model.predict(image)

        cer = get_cer(prediction_text, label)
        print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

        accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")