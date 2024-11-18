# This is the second prototype of the project. This code uses the API token without using .env file which causes security issues. So the cuurent version resolves this issue.

from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import cv2

class Config:
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed = 42
    torch_generator = torch.Generator(device_type).manual_seed(random_seed)
    inference_steps = 35
    diffusion_model_id = "stabilityai/stable-diffusion-2"
    output_image_dimensions = (400, 400)
    guidance_strength = 9
    text_model_id = "gpt2"
    dataset_size = 6
    max_prompt_length = 12

def initialize_diffusion_model():
    """Initialize and return the Stable Diffusion model."""
    dtype = torch.float16 if Config.device_type == "cuda" else torch.float32
    
    diffusion_model = StableDiffusionPipeline.from_pretrained(
        Config.diffusion_model_id,
        torch_dtype=dtype,
        variant="fp16" if Config.device_type == "cuda" else None,
        use_auth_token='api_token'
    )
    diffusion_model = diffusion_model.to(Config.device_type)
    return diffusion_model

def create_image(prompt_text, diffusion_model):
    """Generate an image using the provided text prompt and model."""
    generated_img = diffusion_model(
        prompt_text,
        num_inference_steps=Config.inference_steps,
        generator=Config.torch_generator,
        guidance_scale=Config.guidance_strength
    ).images[0]

    resized_img = generated_img.resize(Config.output_image_dimensions)
    return resized_img

def run_image_generation():
    """Main function to generate and display an image based on a text prompt."""
    set_seed(Config.random_seed)

    diffusion_model = initialize_diffusion_model()

    image_prompt = "Generate an image of a floor plan of a building"
    generated_img = create_image(image_prompt, diffusion_model)

    generated_img.save("generated_output.png")
    plt.imshow(generated_img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_image_generation()