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
#%%
MODEL=Deeplabv3_2
CHECKPOINT_DIR='ckpt3'
LEARNING_RATE=1e-3
IMAGE_PATH='/Users/anhong-eun/Desktop/Pytorch2CoreML/img/m_label4-2_054.png'
SAVE_PATH='deeplab_2'

#%% Load the model (UNet_2)
model = MODEL()
model, optim, st_epoch = load(ckpt_dir=CHECKPOINT_DIR, net=model, optim=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE))
model.eval()

#%% Load a sample image
input_image = Image.open(IMAGE_PATH).convert('RGB')
# input_image.show()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

#%%
with torch.no_grad():
    output = model(input_batch)[0]
torch_predictions = output.argmax(0)
display_segmentation(input_image, torch_predictions)
#%% Wrap the Model to Allow Tracing*
class WrappedUNet_2(nn.Module):
    
    def __init__(self):
        super(WrappedUNet_2, self).__init__()
        self.model =  UNet_2()
        self.model, optim, st_epoch = load(ckpt_dir='ckpt2', net=self.model, optim=torch.optim.Adam(model.parameters(), lr=1e-3))
        self.model.eval()
    def forward(self, x):
        res = self.model(x)
        x = res
        return x

# Trace the Wrapped Model
traceable_model = WrappedUNet_2().eval()
#%%
trace = torch.jit.trace(traceable_model, input_batch)

# Convert the model
mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)

# Save the model without new metadata
name=os.path.join(SAVE_PATH, MODEL)
mlmodel.save(name+"_no_metadata.mlmodel")

# Load the saved model
mlmodel = ct.models.MLModel(name+"_no_metadata.mlmodel")

# Add new metadata for preview in Xcode
labels_json = {"labels": ["background", "teeths"]}

mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)

mlmodel.save(name+"_with_metadata.mlmodel")
# %%
