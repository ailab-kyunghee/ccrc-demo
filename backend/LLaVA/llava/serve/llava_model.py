# this python module is created by Hamza, for IITP Demo.
# This allows the LLaVA model accessible in the server_iitp_version2.py file which is intermediate file between backend and frontend.

import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from PIL import Image
from io import BytesIO
import requests
from transformers import TextStreamer

class LLaVAModel:
    def __init__(self, model_path, load_8bit=False, load_4bit=False, device='cuda'):
        # Initialize the model with required parameters
        self.model_path = model_path
        self.load_8bit = load_8bit
        self.load_4bit = load_4bit
        self.device = device
        disable_torch_init()  # Disable torch init for optimization
        self.tokenizer, self.model, self.image_processor, self.context_len = self.load_model()

    def load_model(self):
        # Load model
        model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_path, None, model_name, self.load_8bit, self.load_4bit, device=self.device)
        return tokenizer, model, image_processor, context_len

    @staticmethod
    def load_image(image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    def generate_response(self, image_file, prompt, temperature=0.2, max_new_tokens=512, debug=False):
        # Set conversation mode based on model
        model_name = get_model_name_from_path(self.model_path)
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            conv_mode = "mistral_instruct"
        elif "v1.6-34b" in model_name.lower():
            conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        conv = conv_templates[conv_mode].copy()

        # Load image
        image = self.load_image(image_file)
        image_size = image.size
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if type(image_tensor) is list:
            image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

        # Prepare input
        if self.model.config.mm_use_im_start_end:
            prompt = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
        else:
            prompt = DEFAULT_IMAGE_TOKEN + '\n' + prompt

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        # Tokenize input
        input_ids = tokenizer_image_token(full_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        # Generate output
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True
            )

        # Decode and return output
        outputs = self.tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if debug:
            print("\n", {"prompt": full_prompt, "outputs": outputs}, "\n")

        return outputs
