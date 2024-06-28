import torch
from base import BaseMiner
from diffusers import (
    AutoPipelineForImage2Image,
    DPMSolverMultistepScheduler,
    DiffusionPipeline,
    DDIMScheduler
)

import threading

from huggingface_hub import hf_hub_download
from neurons.safety import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor
from utils import colored_log, warm_up

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-8steps-CFG-lora.safetensors"

class StableMiner(BaseMiner):
    def __init__(self):
        super().__init__()

        # Load the model
        self.load_models()

        # Optimize model
        self.optimize_models()

        # Serve the axon
        self.start_axon()

        # Start the miner loop
        self.loop()

    def load_models(self):
        # Load the text-to-image model
        print("setting up model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        ## n step lora
        self.pipeline = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(self.device)
        self.pipeline.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
        self.pipeline.fuse_lora()
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config, timestep_spacing="trailing")
        self.steps = 8
        self.guidance_scale = 5

        # Load the image to image model using the same pipeline (efficient)
        # self.i2i_model = AutoPipelineForImage2Image.from_pipe(self.t2i_model).to(
        #     self.config.miner.device,
        # )
        # self.i2i_model.set_progress_bar_config(disable=True)
        # self.i2i_model.scheduler = DPMSolverMultistepScheduler.from_config(
        #     self.i2i_model.scheduler.config
        # )
        self.i2i_model = self.pipeline  # Use the same pipeline for image-to-image tasks
        self.i2i_model.set_progress_bar_config(disable=True)
        self.i2i_model.scheduler = DPMSolverMultistepScheduler.from_config(self.i2i_model.scheduler.config)

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.config.miner.device)
        self.processor = CLIPImageProcessor()

        # Set up mapping for the different synapse types
        self.mapping = {
            "text_to_image": {"args": self.t2i_args, "model": self.pipeline},
            "image_to_image": {"args": self.i2i_args, "model": self.i2i_model},
        }
        print("model setup done")


    def optimize_models(self):
        if self.config.miner.optimize:
            self.t2i_model.unet = torch.compile(
                self.t2i_model.unet, mode="reduce-overhead", fullgraph=True
            )

            # Warm up model
            colored_log(
                ">>> Warming up model with compile... "
                + "this takes roughly two minutes...",
                color="yellow",
            )
            warm_up(self.t2i_model, self.t2i_args)
