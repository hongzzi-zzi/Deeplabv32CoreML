#%%
from PIL import Image
from torchvision import transforms

preprocess = transforms.Compose([
    transforms.RandomAutocontrast(p = 1),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])
tensor2PIL=transforms.ToPILImage()
PIL2tensor=transforms.ToTensor()
#%%
IMAGE_PATH='/Users/anhong-eun/Desktop/Pytorch2CoreML/img/input4-5_064.jpg'

input_image = Image.open(IMAGE_PATH).convert('RGB')
input_image_preprocess = tensor2PIL(preprocess(input_image))
input_image_preprocess.save(IMAGE_PATH.split('.')[0]+'_3.jpg')

# %%
