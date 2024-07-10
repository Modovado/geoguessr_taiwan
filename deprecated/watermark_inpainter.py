"""
Deprecated: find out you don't need to detect and inpaint just do the cropping
but been doing this quite sometime so be keeping this code

watermark inpainters:
1. Stable Diffusion Inpainting with Flash Diffusion LoRA weights [https://github.com/gojasper/flash-diffusion]

2. HINT Inpainting [https://github.com/ChrisChen1023/HINT]
git clone the repo and only call the InpaintingModel from its src

3. Also have tried PixArt-XL with pipeline_pixart_inpaint [https://github.com/PixArt-alpha/PixArt-alpha/pull/131]
and kandinsky-2-2-decoder-inpaint
"""
import argparse
import random
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, LCMScheduler
from diffusers import StableDiffusionInpaintPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
import albumentations as A
from albumentations.pytorch import ToTensorV2
from HINT.src.models import InpaintingModel


image_path: str = r''
mask_path: str = r''
mask_region = 'BR'
crop_size : int = 512
image_size: tuple[int, ...] = (1920, 1080)
steps: int = 4

def seed_everything(SEED: int = 10):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def postprocess(img: torch.Tensor) -> torch.Tensor:
    # [0, 1] => [0, 255]
    img = img * 255.0
    img = img.permute(0, 2, 3, 1)
    return img.int()

def imsave(img, path=None):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)

width, height = image_size

image = Image.open(image_path)
mask = Image.open(mask_path)

minx = max(0, width - crop_size)
miny = max(0, height - crop_size)
maxx = width
maxy = height

orig_image = image

# crop
cropped_image = image.crop((minx, miny, maxx, maxy))
cropped_mask = mask.crop((minx, miny, maxx, maxy))

# IMPAINTING METHOD

# SD-inpainting w/ flash-sd LORA
adapter_id = 'jasperai/flash-sd'

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    'runwayml/stable-diffusion-inpainting',
    variant='fp16',
    torch_dtype=torch.float16,
)
# Fuse and load LoRA weights
pipe.load_lora_weights(adapter_id)
pipe.fuse_lora()

pipe.scheduler = LCMScheduler.from_pretrained(
  "runwayml/stable-diffusion-v1-5",
  subfolder="scheduler",
  timestep_spacing="trailing",
)

pipe.unet.set_attn_processor(AttnProcessor2_0())
pipe.enable_model_cpu_offload()

# torch.compile doesn't work on my old ahh GPU
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)

prompt = ("Generate a realistic street view image from a car's dashcam perspective. "
          "Seamlessly blend it with the provided dashcam photo, "
          "removing any logos or unwanted elements and replacing them with plausible content. "
          "Ensure the result appears naturally captured by a dashcam, "
          "maintaining consistency with the surrounding environment in terms of lighting, "
          "perspective, and overall scene composition.")

inpainted_image = pipe(prompt,
                       image=cropped_image,
                       mask_image=cropped_mask,
                       num_inference_steps=steps).images[0]

orig_image.paste(inpainted_image, (minx, miny))
orig_image.save('fixed.jpg')
########################################################################################################################
# HINT-inpainting

np_cropped_image = np.array(cropped_image)
np_cropped_mask = np.array(cropped_mask)

# config
DEFAULT_CONFIG = {
    'MODE': 2,  # 1: train, 2: test, 3: eval
    'MODEL': 2,  # 1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model
    'MASK': 6,  # 1: random block, 2: half, 3: external, 4: (external, random block), 5: (external, random block, half)
    'NMS': 1,  # 0: no non-max-suppression, 1: applies non-max-suppression on the external edges by multiplying by Canny
    'SEED': 10,  # random seed
    'GPU': [0],  # list of gpu ids
    'AUGMENTATION_TRAIN': 0,  # 1: train 0: false use augmentation to train landmark predictor

    'LR': 0.0001,  # learning rate
    'D2G_LR': 0.1,  # discriminator/generator learning rate ratio
    'BETA1': 0.9,  # adam optimizer beta1
    'BETA2': 0.999,  # adam optimizer beta2
    'WD': 0,
    'LR_Decay': 1,

    'BATCH_SIZE': 4,  # input batch size for training
    'INPUT_SIZE': 256,  # input image size for training 0 for original size
    'MAX_ITERS': 300001,  # maximum number of iterations to train the model

    'L1_LOSS_WEIGHT': 1,  # l1 loss weight
    'STYLE_LOSS_WEIGHT': 250,  # style loss weight
    'CONTENT_LOSS_WEIGHT': 0.1,  # perceptual loss weight
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,  # adversarial loss weight

    'TV_LOSS_WEIGHT': 0.1,  # total variation loss weight

    'GAN_LOSS': 'lsgan',  # nsgan | lsgan | hinge
    'GAN_POOL_SIZE': 0,  # fake images pool size

    'SAVE_INTERVAL': 1000,  # how many iterations to wait before saving model (0: never)
    'SAMPLE_INTERVAL': 1000,  # how many iterations to wait before sampling (0: never)
    'SAMPLE_SIZE': 12,  # number of images to sample
    'EVAL_INTERVAL': 0,  # how many iterations to wait before model evaluation (0: never)
    'LOG_INTERVAL': 100,  # how many iterations to wait before logging training status (0: never)

    'PATH': r'D:\\Python_Projects\\Geoguessr_tw\\HINT\\checkpoints\\',
    'DEVICE': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
}

parser = argparse.ArgumentParser()
parser.set_defaults(**DEFAULT_CONFIG)
config = parser.parse_args()
# build the model and initialize

model = InpaintingModel(config)
model.load()
model.eval()

# cuda
model = model.cuda()


image_transform = A.Compose([
    A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
    ToTensorV2(),
])

mask_transform =  A.Compose([
    A.Normalize(mean=0, std=1),
    ToTensorV2(),
])
# transformed = transform(image=np_cropped_image, mask=np_cropped_mask)
# inputs, masks = transformed["image"], transformed["mask"]
image_transformed = image_transform(image=np_cropped_image)
mask_transformed = mask_transform(image=np_cropped_mask)

images, masks = image_transformed["image"], mask_transformed["image"]

# add batch_dim
images = torch.unsqueeze(images, 0)
masks = torch.unsqueeze(masks, 0)

# float & cuda
images = images.float().cuda()
masks = masks.float().cuda()


# PRE-PROCESS
inputs = (images * (1 - masks))

with torch.no_grad():
    outputs = model(inputs, masks)

# POST-PROCESS
# outputs_merged = (outputs * masks) + (images * (1 - masks))
outputs_merged = (outputs * masks) + (inputs * (1 - masks))

inpainted_image = postprocess(outputs_merged)
imsave(inpainted_image, 'inpainted.png')

inputs_image = postprocess(inputs)
imsave(inputs_image, 'inputs.png')

masks_image = postprocess(masks)
imsave(masks_image, 'masks.png')
########################################################################################################################
# kandinsky
# pipeline = AutoPipelineForInpainting.from_pretrained(
#     "kandinsky-community/kandinsky-2-2-decoder-inpaint",
#     torch_dtype=torch.float16,
    # use_safetensors=True
# )
