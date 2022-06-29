#%% import package
import urllib
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import json

import coremltools as ct
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

from model import Deeplabv3_2, UNet_1, UNet_2
from util import *

#%%
MODEL=UNet_1
CHECKPOINT_DIR='ckpt_unet'
LEARNING_RATE=1e-3
IMAGE_PATH='/Users/anhong-eun/Desktop/Pytorch2CoreML/img/input1_029.jpg'
SAVE_PATH='UNet'

teeth_class=lambda x:1.0*(x>0.5)
bg_class=lambda x:1.0*(x<=0.5)
tensor2PIL=transforms.ToPILImage()

#%% Load the model (UNet_1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ,라뷰류ㅏ규2구  
model = MODEL()
model, optim, st_epoch = load(ckpt_dir=CHECKPOINT_DIR, net=model, optim=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE))
model.eval()

#%% Load a sample image
input_image = Image.open(IMAGE_PATH).convert('RGB').resize((512, 512))
input_image.show()

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

#%%
with torch.no_grad():
    output = model(input_batch)[0]
#%%
torch_predictions1 = bg_class(output)
torch_predictions2 = teeth_class(output)
#%%
# torch_predictions= torch_predictions1.squeeze()
torch_predictions=torch.concat([torch_predictions1, torch_predictions2], dim=0)
display_segmentation(input_image, torch_predictions.argmax(0))
# #%%
# display_segmentation(input_image, torch_predictions1[0])
# display_segmentation(input_image, torch_predictions2[0])
#%%
# print(output)
#%% Wrap the Model to Allow Tracing*
class WrappedUNet_1(nn.Module):
    
    def __init__(self):
        super(WrappedUNet_1, self).__init__()
        self.model =  UNet_1
        self.model, optim, st_epoch = load(ckpt_dir='ckpt_unet', net=model, optim=torch.optim.Adam(model.parameters(), lr=1e-3))
        self.model.eval()
    def forward(self, x):
        res = self.model(x)[0]
        # xx=fn_class(res).unsqueeze(0)
        bg = bg_class(res)
        teeth = teeth_class(res)
        x=torch.concat([bg, teeth], dim=0).unsqueeze(0)

        return x

# Trace the Wrapped Model
traceable_model = WrappedUNet_1().eval()

#%%
trace = torch.jit.trace(traceable_model, input_batch)
  #%%
# Convert the model
mlmodel = ct.convert(
    trace,
    inputs=[ct.TensorType(name="input", shape=input_batch.shape)],
)
#%%
# Save the model without new metadata
name=os.path.join(SAVE_PATH,str(MODEL))
mlmodel.save(name+"_no_metadata.mlmodel")

# Load the saved model
mlmodel = ct.models.MLModel(name+"_no_metadata.mlmodel")

# Add new metadata for preview in Xcode
labels_json = {"labels": ["background", "teeths"]}

mlmodel.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageSegmenter"
mlmodel.user_defined_metadata['com.apple.coreml.model.preview.params'] = json.dumps(labels_json)
mlmodel.save(name+"_with_metadata.mlmodel")

#%%
# # %%
# outputimg=tensor2PIL(fn_class(output[0]))
# outputimg.show()
# %%
# print(output[0].shape)# torch.Size([512, 512])
# print(output)#)# torch.Size([1, 512, 512])
# %%
