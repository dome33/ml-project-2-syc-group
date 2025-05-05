import torch
from models.cnn_owen import HandwritingRecognitionCNN_BiLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dummy_input = torch.randn(1, 64, 256)  # Example for an image model

num_classes = 30
hidden_size = 64

model_path = "results/falconet/model_0_14870.pt"

model = HandwritingRecognitionCNN_BiLSTM(num_classes=num_classes, hidden_size=hidden_size)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

torch.onnx.export(
    model,                      # Your PyTorch model
    dummy_input,                # Example input
    "falconet_model.onnx",               # Output file name
    export_params=True,         # Store the trained weights
    opset_version=11,           # ONNX version (11 is widely supported)
    do_constant_folding=True,   # Optimize constant expressions
    input_names=['input'],      # Name of the input node
    output_names=['output'],    # Name of the output node
    dynamic_axes={              # Optional: for dynamic batch size
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

