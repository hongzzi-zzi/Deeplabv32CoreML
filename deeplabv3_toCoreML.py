#%% import package
import urllib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torchvision
import json
from util import *

from torchvision import transforms
from PIL import Image

import coremltools as ct

#%% Load the model (deeplabv3)
model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).eval()

# Load a sample image (cat_dog.jpg)
input_image = Image.open("img/cat_dog.jpeg")
# input_image.show()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5,)
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_batch)['out'][0]
torch_predictions = output.argmax(0)


display_segmentation(input_image, torch_predictions)

#%% Wrap the Model to Allow Tracing*
class WrappedDeeplabv3Resnet101(nn.Module):
    
    def __init__(self):
        super(WrappedDeeplabv3Resnet101, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).eval()
    
    def forward(self, x):
        res = self.model(x)
        x = res["out"]
        return x
        
# Trace the Wrapped Model
traceable_model = WrappedDeeplabv3Resnet101().eval()
trace = torch.jit.trace(traceable_model, input_batch)

# Convert the model
mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)

# Save the model without new metadata
mlmodel.save("deeplab/SegmentationModel_no_metadata.mlmodel")

# Load the saved model
mlmodel = ct.models.MLModel("deeplab/SegmentationModel_no_metadata.mlmodel")

# Add new metadata for preview in Xcode
labels_json = {"labels": ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]}

mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)

mlmodel.save("deeplab/SegmentationModel_with_metadata.mlmodel")
# %%
