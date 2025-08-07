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
        input_image: Path = Input(description="Uploaded photo to convert to line art")
    ) -> Path:
        torch.manual_seed(42)

        image = Image.open(input_image).convert("RGB")
        edge = self.hed(image).resize(image.size)

        prompt = (
            "Ultra sharp black and white line art drawing of the subject and background, "
            "no color, no shading, no gray areas, highly detailed, clean outlines, crisp edges, "
            "full scene, adult coloring book page style, extremely accurate, realistic proportions."
        )

        result = self.pipe(
            prompt=prompt,
            image=edge,
            num_inference_steps=30,
            guidance_scale=13.5
        ).images[0]

        output_path = "/tmp/output.png"
        result.save(output_path)
        return Path(output_path)
