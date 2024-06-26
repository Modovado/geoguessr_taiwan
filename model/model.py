
# helper function
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
'''
PIGEON for Geoguessr-tw

PIGEON uses CLIP ViT-L/14 336
Clip-ViT -> text-encoder and image-encoder
LoRA from PEFT by HuggingFace

'''
import torch
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# Load the CLIP model and processor
# model_name = "openai/clip-vit-base-patch32"
model_name = "openai/clip-vit-large-patch14-336"
model = CLIPModel.from_pretrained(model_name)
'''
CLIPProcessor wraps CLIPImageProcessor and CLIPTokenizer into a single instance  to both encode the text and prepare the 
images.
'''
processor = CLIPProcessor.from_pretrained(model_name)

# Define LoRA configuration for text encoder and image encoder
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=16,  # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "value", "fc1", "fc2"],  # Modules to apply LoRA to # all
    lora_dropout=0.1,  # Dropout rate
    bias="none"  # Whether to include bias terms
)

# Apply LoRA to the image encoder
model.vision_model = get_peft_model(model.vision_model, lora_config)

print_trainable_parameters(model.vision_model)

# Apply LoRA to the text encoder
model.text_model = get_peft_model(model.text_model, lora_config)

print_trainable_parameters(model.text_model)

#
# # Synthetic Captions
#
# label = ''
#
# caption = ''



if __name__ == '__main__':

    print('ForsenCD')
    # print(model.vision_model)
    # print(model.text_model)

