import torch
import numpy as np
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from PIL import Image
from controlnet_aux import HEDdetector

class Predictor(BasePredictor):
    def setup(self):
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-hed",
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()

        self.hed = HEDdetector.from_pretrained("lllyasviel/ControlNet")

    def predict(
        self,
        input_image: Path = Input(description="Uploaded photo to convert to line art"),
        prompt: str = Input(
            description="Prompt for style",
            default=(
                "High-resolution black and white line art of the uploaded photo. "
                "Capture all people and full background exactly as shown, using smooth, clean outlines with consistent thickness. "
                "No shading, no sketch lines, no distortion or abstraction. "
                "Facial features and proportions must be precise. "
                "Maintain real-life accuracy so the image is instantly recognizable. "
                "Final result should resemble a professional adult coloring book page."
            )
        ),
        num_inference_steps: int = Input(description="Steps for generation", default=30),
        guidance_scale: float = Input(description="Classifier-free guidance scale", default=13.5),
        seed: int = Input(description="Random seed (for reproducibility)", default=12345),
    ) -> Path:
        torch.manual_seed(seed)

        image = Image.open(input_image).convert("RGB")
        edge = self.hed(image).resize(image.size)

        result = self.pipe(
            prompt=prompt,
            image=edge,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        output_path = "/tmp/output.png"
        result.save(output_path)
        return Path(output_path)
