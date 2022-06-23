#%% import package
import urllib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torchvision
import json
from model import UNet_1, UNet_2, Deeplabv3_2
from torchvision import transforms
from PIL import Image
from util import *
import coremltools as ct

#%% Load the model (deeplabv3)
# model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).eval()
model = UNet_1()
model, optim, st_epoch = load(ckpt_dir='ckpt', net=model, optim=torch.optim.Adam(model.parameters(), lr=1e-3))
model.eval()

#%% Load a sample image
input_image = Image.open("/Users/anhong-eun/Desktop/Pytorch2CoreML/img/m_label1_001.png").convert('RGBA')
# input_image.show()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

input_tensor = preprocess(input_image)
input_batch = input_tensor#.unsqueeze(0)

#%%
with torch.no_grad():
    output = model(input_batch)#['out'][0]
print(output)
print(output.shape)# torch.Size([21, 448, 448])
print(output.argmax(0).shape)# torch.Size([448, 448])
print(output.argmax(0))
torch_predictions = output.argmax(0)
print(torch_predictions.shape)
#%%
def display_segmentation(input_image, output_predictions):
    # Create a color pallette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # Plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(
        output_predictions.byte().cpu().numpy()
    ).resize(input_image.size)
    r.putpalette(colors)

    # Overlay the segmentation mask on the original image
    alpha_image = input_image.copy()
    alpha_image.putalpha(255)
    r = r.convert("RGBA")
    r.putalpha(128)
    seg_image = Image.alpha_composite(alpha_image, r)
    # display(seg_image) -- doesn't work
    seg_image.show()

display_segmentation(input_image, torch_predictions)

# Wrap the Model to Allow Tracing*
class WrappedDeeplabv3Resnet101(nn.Module):
    
    def __init__(self):
        super(WrappedDeeplabv3Resnet101, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).eval()
    
    def forward(self, x):
        res = self.model(x)
        x = res["out"]
        return x
#%%
# Trace the Wrapped Model
traceable_model = WrappedDeeplabv3Resnet101().eval()
trace = torch.jit.trace(traceable_model, input_batch)

# Convert the model
mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)

# Save the model without new metadata
# mlmodel.save("SegmenstationModel_no_metadata.mlmodel")

# Load the saved model
mlmodel = ct.models.MLModel("UNet1_no_metadata.mlmodel")

# Add new metadata for preview in Xcode
# labels_json = {"labels": ["background", "aeroplane", "bicycle", "bird", "board", "bottle", "bus", "car", "cat", "chair", "cow", "diningTable", "dog", "horse", "motorbike", "person", "pottedPlant", "sheep", "sofa", "train", "tvOrMonitor"]}
labels_json = {"labels": ["background", "teeths"]}

mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)

mlmodel.save("SegmentationModel_with_metadata.mlmodel")
# %%
