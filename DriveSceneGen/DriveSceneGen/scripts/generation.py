from diffusers import DDPMPipeline
import os

# initialize the model
diffusion_steps = 1000
modelpath = "/data/haibin/ML_DM/model/train_20s_A100"
ddpm = DDPMPipeline.from_pretrained(modelpath,variant="fp16").to('cuda')

# output_dir = "/data/haibin/ML_DM/generation/generated_80m_5k/diffusion_good"
output_dir = "/data/haibin/ML_DM/genernation/20s_A100_2000/"
os.makedirs(output_dir, exist_ok=True)

for num in range(100):
    # generate dx dy
    polylines = ddpm(
        batch_size = 36,
        # generator=torch.manual_seed(1),
        num_inference_steps=diffusion_steps,
        #   output_type="pil",
        #   return_dict=False
        ).images

    if num > 5:

        for i, image in enumerate(polylines):
            # save image
            image.save(f"{output_dir}/loop_{num:03d}_batch_{i:03d}.png")
