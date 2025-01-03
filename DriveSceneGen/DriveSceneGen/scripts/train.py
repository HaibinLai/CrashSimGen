import torch
from diffusers import UNet2DModel,DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import notebook_launcher
import matplotlib.pyplot as plt
from DriveSceneGen.utils.datasets.dataset import Image_Dataset
from DriveSceneGen.pipeline.training_pipeline import TrainingPipeline
from dataclasses import dataclass
from torchvision import transforms

####---initial config---####
@dataclass
class TrainingConfig:
    patterns_size_height = 256 # max 400 remember change at dataset ##: 62401 pictures
    patterns_size_width = 256
    train_batch_size = 36  # train batch size
    eval_batch_size = 1 
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-5  # change
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 1 # save model epoch
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = '/data/haibin/ML_DM/model/1_new'  # the generated model name
    # dataset_name = "/data/haibin/ML_DM/rasterized/GT_70k_s80_dxdy_agents_img/*"
    dataset_name = "/data/haibin/ML_DM/rasterized_training_20s/1_new/*"
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 14555

config = TrainingConfig()

####---load dataset---####

dataset = Image_Dataset(config)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

##---load model---####

model = UNet2DModel(
    sample_size=(config.patterns_size_height,config.patterns_size_width),  # the target pattern resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 256, 512),  # the number of output channes for each UNet block
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",  
        "DownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ), 
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "UpBlock2D",  # a ResNet upsampling block with spatial self-attention  
        "UpBlock2D",
        "UpBlock2D"  
    ),
)

# model = UNet2DModel.from_pretrained(config.output_dir,subfolder="unet")
print("model parameters:",sum(p.numel() for p in model.parameters() if p.requires_grad))

####---load optimizer, scheduler and pipeline---####

pipeline = TrainingPipeline(config, inference_steps=1000)
noise_scheduler = DDPMScheduler()
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def add_noise_verification(data,index,noisy=True,intensities=100):
    
    inverse_normalize = transforms.Compose([
        # transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    ])
    
    toimage = inverse_normalize(data[index])
     
    if noisy == True:
        
        #add noise to original pattern        
        noise = torch.randn(toimage.shape)

        #diffusion timesteps, determine how much noise to add
        timesteps = torch.LongTensor([intensities])
        
        #add noise to original pattern
        noisy_pattern = noise_scheduler.add_noise(toimage, noise, timesteps)
        
        #tensor->numpy
        numpy_noisy_pattern = noisy_pattern.numpy()
    
        plt.imshow(numpy_noisy_pattern[:,:,:])
    else:
        
        # additional_channel=torch.ones(toimage.shape[0],toimage.shape[1],1)*0.5
        # toimage = torch.cat((toimage,additional_channel),dim=2)
        # plt.imshow(toimage[:,:,2].numpy(),cmap='gray')
        
        plt.imshow(toimage[:,:,:])
    
    plt.show()








def training_mine(NUM_EPOCHS=50, INFER_STEPS=1000, TRAIN_BATCH_SIZE=36):

    # ##---load dataset---####
    @dataclass
    class TrainingConfig_1:
        patterns_size_height = 256 # max 400 remember change at dataset ##: 62401 pictures
        patterns_size_width = 256
        train_batch_size = 36  # train batch size
        eval_batch_size = 1 
        num_epochs = 50
        gradient_accumulation_steps = 1
        learning_rate = 1e-5  # change
        lr_warmup_steps = 500
        save_image_epochs = 1
        save_model_epochs = 1 # save model epoch
        mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
        output_dir = '/data/haibin/ML_DM/model/fine_tune'  # the generated model name
        # dataset_name = "/data/haibin/ML_DM/rasterized/GT_70k_s80_dxdy_agents_img/*"
        dataset_name = "/data/haibin/ML_DM/rasterized_training_20s/1_1_new/*"
        overwrite_output_dir = True  # overwrite the old model when re-running the notebook
        seed = 14555

        def __init__(self, learning_rate: float, num_epochs: int, train_batch_size: int):
            # Set custom logic or override default values
            self.learning_rate = learning_rate
            self.num_epochs = num_epochs
            # Optional: Set default values for other fields
            self.patterns_size_height = 256
            self.patterns_size_width = 256
            self.train_batch_size = train_batch_size
            self.eval_batch_size = 1
            self.gradient_accumulation_steps = 1
            self.lr_warmup_steps = 500
            self.save_image_epochs = 1
            self.save_model_epochs = 1
            self.mixed_precision = 'fp16'
            self.output_dir = '/data/haibin/ML_DM/model/fine_tune'
            self.dataset_name = "/data/haibin/ML_DM/rasterized_training_20s/1_1_new/*"
            self.overwrite_output_dir = True
            self.seed = 14555


    config_1 = TrainingConfig_1(learning_rate=1e-5, num_epochs=NUM_EPOCHS, train_batch_size=TRAIN_BATCH_SIZE)

    Mydataset = Image_Dataset(config_1)
    train_dataloader = torch.utils.data.DataLoader(Mydataset, batch_size=config_1.train_batch_size, shuffle=True)

    print("dataset:",len(Mydataset))

    init_channel = 64
    init_channel2 = 128
    init_channel3 = 256
    init_channel4 = 512

    ## UNet2D Model
    model_1 = UNet2DModel(
        sample_size=(config_1.patterns_size_height,config_1.patterns_size_width),  # the target pattern resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(init_channel, init_channel2, init_channel3, init_channel4),  # the number of output channes for each UNet block
        down_block_types=( 
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",  
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention  
            "UpBlock2D",
            "UpBlock2D"  
        ),
    )
    # model = UNet2DModel.from_pretrained(config.output_dir,subfolder="unet")
    print("model parameters:",sum(p.numel() for p in model_1.parameters() if p.requires_grad))


    pipeline = TrainingPipeline(config_1, inference_steps=INFER_STEPS)
    noise_scheduler = DDPMScheduler()
    # noise_scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
    optimizer = torch.optim.AdamW(model_1.parameters(), lr=config_1.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config_1.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config_1.num_epochs),
    )



    ## train
    args = (config_1, model_1, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(pipeline.train_loop, args, num_processes=1)
    









####---start training---####

if __name__ == "__main__":
         
    print("dataset:",len(dataset))
    
    ## show examples, with forward diffusion
    # for example_index in range(0,len(dataset)):
    #     print(dataset[example_index].shape)
    #     add_noise_verification(dataset, example_index, noisy=False, intensities=100)
    
    ## train
    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    notebook_launcher(pipeline.train_loop, args, num_processes=1)

