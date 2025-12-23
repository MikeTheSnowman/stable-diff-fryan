from cog import BasePredictor, Path
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
import os
from dotenv import load_dotenv

import tempfile

# Load environment variables
load_dotenv()

class Predictor(BasePredictor):
    
    def setup(self):
        self.model_id = "./stable-diffusion-3.5-medium"

        self.nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model_nf4 = SD3Transformer2DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            quantization_config=self.nf4_config,
            torch_dtype=torch.bfloat16,
            token=os.getenv("HUGGINGFACE_TOKEN")
        )

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.model_id,
            transformer=self.model_nf4,
            torch_dtype=torch.bfloat16
        )

        self.pipeline.enable_model_cpu_offload()

    def predict(self, prompt: str) -> Path:
        
        image = self.pipeline(
            prompt=prompt, 
            num_inference_steps=40, 
            guidance_scale=4.5, 
            max_sequence_length=512).images[0]
        

        output_path = Path(tempfile.mkdtemp()) / "upscaled.png"
        image.save(output_path)
        return Path(output_path)
