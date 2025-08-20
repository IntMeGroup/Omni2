import os
import datasets
from datasets import load_dataset, ClassLabel, concatenate_datasets
import torch
import numpy as np
import random
from PIL import Image
import json
import copy
# import torchvision.transforms as T
from torchvision import transforms
import pickle 
import re

from OmniGen import OmniGenProcessor
from OmniGen.processor import OmniGenCollator
from preprocess_360 import Equirec2Perspec as E2P
import cv2
class DatasetFromJson(torch.utils.data.Dataset):
    def __init__(
        self,
        json_file: str, 
        image_path: str,
        processer: OmniGenProcessor,
        image_transform,
        max_input_length_limit: int = 18000,
        condition_dropout_prob: float = 0.1,
        keep_raw_resolution: bool = True, 
    ):
        
        self.image_transform = image_transform
        self.processer = processer
        self.condition_dropout_prob = condition_dropout_prob
        self.max_input_length_limit = max_input_length_limit
        self.keep_raw_resolution = keep_raw_resolution

        self.data = load_dataset('json', data_files=json_file)['train']
        self.image_path = image_path

    def process_360_image(self, image_file, InterpolationMode=False):

        if InterpolationMode:
            bicubic = InterpolationMode.BICUBIC
        else:
            bicubic = Image.BICUBIC
        if self.image_path is not None:
            image_file = os.path.join(self.image_path, image_file)
        equ = E2P.Equirectangular(image_file)

        output_size = 256

        views = {
            'up': (100, 0, 90),
            'front': (100, 0, 0),
            'left': (100, 90, 0),
            'back': (100, 180, 0),
            'right': (100, 270, 0),
            'down': (100, 0, -90),
            }
        
        perspective_images_processed = []
        for view_name, (fov, theta, phi) in views.items():
            img = equ.GetPerspective(fov, theta, phi, output_size, output_size)
            # img_pil = Image.fromarray(img) 
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb) 
            img_processed = self.image_transform(img_pil)
            perspective_images_processed.append(img_processed) #len=6 3-dim tensor inside

        return perspective_images_processed
    
    def process_image(self, image_file):
        if self.image_path is not None:
            image_file = os.path.join(self.image_path, image_file)
        image = Image.open(image_file).convert('RGB')
        return self.image_transform(image)

    def get_example(self, index):
        example = self.data[index]
        
        instruction, input_images, output_image = example['instruction'], example['input_images'], example['output_image']
        if random.random() < self.condition_dropout_prob:
            '''for generation tasks: note that this could lead to data-missing in editing tasks'''
            instruction = '<cfg>'
            input_images = None
        if input_images is not None:
            # input_images = [self.process_image(x) for x in input_images]
            input_images = [self.process_360_image(x) for x in input_images] #len=1 len=6 [3, 512, 512]
        mllm_input = self.processer.process_multi_modal_prompt(instruction, input_images)
        #mllm_input=['input_ids', 'pixel_values','image_sizes']
        #pixel_values: len=1 len=6 [3,512,512]
        #image_sizes:len=1 [15,6159](contains the start and end index of the image tokens)
        output_image = self.process_360_image(output_image)
            
        return (mllm_input, output_image)


    def __getitem__(self, index):
        return self.get_example(index)
        for _ in range(8):
            try:
                mllm_input, output_image = self.get_example(index)
                if len(mllm_input['input_ids']) > self.max_input_length_limit:
                    raise RuntimeError(f"cur number of tokens={len(mllm_input['input_ids'])}, larger than max_input_length_limit={self.max_input_length_limit}")
                return mllm_input, output_image
            except Exception as e:
                print("error when loading data: ", e)
                print(self.data[index])
                index = random.randint(0, len(self.data)-1)
        raise RuntimeError("Too many bad data.")
    

    def __len__(self):
        return len(self.data)



class TrainDataCollator(OmniGenCollator):
    def __init__(self, pad_token_id: int, hidden_size: int, keep_raw_resolution: bool):
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.keep_raw_resolution = keep_raw_resolution

    def __call__(self, features):
        mllm_inputs = [f[0] for f in features] #bs=2 (mllm_input, output_images)
        # for i in range(6): #6 viewports
        # output_images = [f[1].unsqueeze(0) for f in features]
        output_images = [[tensor.unsqueeze(0) for tensor in f[1]] for f in features] #[bs, view, [1,3,512,512]]
        # print(len(output_images)) 
        # print(len(output_images[1]))
        # print(output_images[0][0].shape)
        # target_img_size = [[x.size(-2), x.size(-1)] for x in output_images] #[[512, 1024], [512, 1024]]
        target_img_size = [[[tensor.size(-2), tensor.size(-1)] for tensor in sublist] for sublist in output_images] #len=bs tensor[[512,512], [512,512]]...(6)
        all_padded_input_ids, all_position_ids, all_attention_mask, all_padding_images, all_pixel_values, all_image_sizes = self.process_mllm_input(mllm_inputs, target_img_size)

        if not self.keep_raw_resolution: #self.keep_raw_resolution = True
            output_images = torch.cat(output_images, dim=0) #concate in bs dimension
            if len(all_pixel_values) > 0:
                all_pixel_values = torch.cat(all_pixel_values, dim=0)
            else:
                all_pixel_values = None

        data = {"input_ids": all_padded_input_ids,
        "attention_mask": all_attention_mask,
        "position_ids": all_position_ids, #[bs, len_of_squence]
        "input_pixel_values": all_pixel_values, #len=bs len=6 [1,3,512,512]
        "input_image_sizes": all_image_sizes,
        "padding_images": all_padding_images,
        "output_images": output_images,  #len=bs len=6 [1,3,512,512]
        }
        return data





