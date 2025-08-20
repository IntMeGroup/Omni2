from pipeline import Omni2Pipeline
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

pipe = Omni2Pipeline.from_pretrained("/DATA/DATA1/yangliu/pretrained_models/OmniGen-v1")  
print('successfully loaded original weights!') 
pipe.merge_lora("/DATA/DATA3/yl/Omni2_code/omni2/omni-edit/results/all_in_one_bs16_device_2_lora_16/checkpoints/0012000") 
print('lora weights loaded!') 

# Text to ODI
images = pipe( 
    prompt="your text descroption here.", 
    height=256, 
    width=256, 
    num_inference_steps=50, 
    guidance_scale=3.0, 
    seed=66, 
) 
os.makedirs("test/t2i", exist_ok=True)
images[0].save("test/t2i/0.png")
images[1].save("test/t2i/1.png")
images[2].save("test/t2i/2.png")
images[3].save("test/t2i/3.png")
images[4].save("test/t2i/4.png")
images[5].save("test/t2i/5.png")

# ## Multi-modal to ODI
# start_time = time.time()
# # '''outpainting'''
# images = pipe( 
#     prompt="<img><|image_1|></img> outpaint", 
#     input_images=["path to your masked image."], 
#     height=256, 
#     width=256, 
#     num_inference_steps=50, 
#     guidance_scale=2.5, 
#     img_guidance_scale=1.8, 
#     seed=3600
# ) 

# os.makedirs("test/outpainting", exist_ok=True) 
# images[0].save("test/outpainting/0.png")  # save output PIL Image 
# images[1].save("test/outpainting/1.png") 
# images[2].save("test/outpainting/2.png") 
# images[3].save("test/outpainting/3.png") 
# images[4].save("test/outpainting/4.png") 
# images[5].save("test/outpainting/5.png") 
# end_time = time.time()
# print(end_time - start_time)