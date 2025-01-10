import ddpm 
from PIL import Image
from torchvision import transforms
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

image = Image.open("giraffe.jpg")

image_size = 128
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


x_start = transform(image).unsqueeze(0)

diffusion_linear = ddpm.Diffusion(noise_steps=500)
diffusion_cosine = ddpm.Diffusion(noise_steps=500,beta_schedule='cosine')

plt.figure(figsize=(16, 8))
for idx, t in enumerate([0, 50, 100, 200, 499]): 
    x_noisy,_ = diffusion_linear.q_sample(x_start, t=torch.tensor([t])) # 使用q_sample去生成x_t
    x_noisy2,_ = diffusion_cosine.q_sample(x_start,t=torch.tensor([t])) # [1,3,128,128]
    noisy_image = (x_noisy.squeeze().permute(1, 2, 0) + 1) * 127.5  # 我们的x_t被裁剪到（-1，1）,所以+1后乘以127.5
    noisy_img2 = (x_noisy2.squeeze().permute(1,2,0)+1)*127.5 # # [128,128,3] -> (0,2) 
    noisy_image = noisy_image.numpy().astype(np.uint8)
    noisy_img2 = noisy_img2.numpy().astype(np.uint8)
    plt.subplot(2, 5, 1 + idx)
    plt.imshow(noisy_image)
    plt.axis("off")
    plt.title(f"t={t}")
    plt.subplot(2, 5, 6+idx)
    plt.imshow(noisy_img2)
    plt.axis('off')
plt.figtext(0.5, 0.95, 'Linear Beta Schedule', ha='center', fontsize=16)  # 在第一行上方添加大标题
plt.figtext(0.5, 0.48, 'Cosine Beta Schedule', ha='center', fontsize=16)  # 在第二行上方添加大标题
plt.savefig('temp_img/add_noise_process.png')



class SelfAttention(nn.Module):
	def __init__(self,channels):
		super().__init__()
		self.channels = channels
		self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
		self.ln = nn.LayerNorm([channels])
		self.ff = nn.Sequential(
			nn.LayerNorm([channels]),
			nn.Linear(channels,channels),
			nn.GELU(),
			nn.Linear(channels,channels)
			)
	
	def forward(self,x):
		B,C,H,W = x.shape
		x = x.reshape(-1,self.channels,H*W).swapaxes(1,2)
		x_ln = self.ln(x)
		attention_value = self.mha(x_ln)
		attention_value = attention_value + x
		attention_value = self.ff(attention_value)+ attention_value
		return attention_value.swapaxes(1,2).view(-1,self.channels,H,W)


class DoubleConv(nn.Module):
	def __init__(self,in_c,out_c,mid_c=None,residual=False):
		super().__init__()
		self.residual = residual
		if mid_c is None:
			mid_c = out_c
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_c,mid_c,kernel_size=3,padding=1),
			nn.GroupNorm(1,mid_c),
			nn.GELU(),
			nn.Conv2d(mid_c,out_c,kernel_size=3,padding=1),
			nn.GroupNorm(1,mid_c)
			)
		if in_c != out_c:
			self.shortcut = nn.Conv2d(in_c,out_c,kernel_size=1)
		else:
			self.shortcut = nn.Identity()
	
	def forward(self,x):
		if self.residual:
			return F.gelu(self.shortcut(x)+self.double_conv(x))
		else:
			return F.gelu(self.double_conv(x))



class Down(nn.Module):
	def __init__(self,in_c,out_c,emb_dim=256):
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2) # kernel_size=2, stride default equal to k
			DoubleConv(in_c,out_c,residual=True),
			DoubleConv(out_c,out_c)
			)
		self.emb_layer = nn.Sequential(
			nn.SiLU(),
			nn.Linear(emb_dim,out_c)
			)
	
	def forward(self,x,t):
		x = self.maxpool_conv(x)
		emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
		# 扩维后，在最后两维重复h和w次，此时和x的尺寸相同
		return x+emb


class Up(nn.Module):
	def __init__(self,in_c,out_c,emb_dim=256):
		self.up =  nn.UpSample(scale_factor=2,mode='bilinear', align_corner=True)
		self.conv = nn.Sequential(
			nn.Conv2d(in_c,in_c,residual=True),
			nn.Conv2d(in_c,out_c)
			)
		self.emb_layer = nn.Sequential(
			nn.SiLU(),
			nn.Linear(emb_dim,out_c)
			)
	
	def forward(self,x,skip_x, t):
		x = self.up(x)
		x = torch.cat([x,skip_x],dim=1)
		x = self.conv(x)
		emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1])
		return x + emb




class UNet(nn.Module):
	def __init__(self,in_c, out_c, time_dim=256, device='cuda'):
	    super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 512)
        self.sa3 = SelfAttention(512)

        self.bot1 = DoubleConv(512, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
	
	def pos_encoding(self,t,channels):
		freq = 1.0/(10000**torch.arange(0,channels,2,device=self.device).float()/channels)
		args = t[:,None].float()*freq[None]
		embedding = torch.cat([torch.sin(args), torch.cos(args)],dim=-1)
		if channels % 2 != 0:
			embedding = torch.cat([embedding,torch.zeros_like(embedding[:,:1])],dim=-1)
		return embeddig
	
	def forward(self,x,t):
		t = self.pos_encoding(t,self.time_dim)
		x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


    def linear_beta_schedule(self):
        scale = 1000/self.noise_steps
        beta_start = self.beta_start*scale
        beta_end = self.beta_end*scale
        return torch.linspace(beta_start, beta_end, self.noise_steps)

    def cosine_beta_schedule(self,s=0.008):
        """
        as proposed in Improved ddpm paper;
		"""
        steps = self.noise_steps + 1
        x = torch.linspace(0, self.noise_steps, steps, dtype=torch.float64) # 从0到self.noise_steps
        alphas_cumprod = torch.cos(((x / self.noise_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]) # alpha_cumprod包含了noise_steps+1个值，则alpha_t是第一个到最后一个；alpha_{t-1}是第0个到倒数第二个（第0个为0）
        return torch.clip(betas, 0, 0.999) # 不大于0.999


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, beta_schedule='linear',device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        if beta_schedule == 'linear':
            self.beta = self.linear_beta_schedule().to(device)
        elif beta_schedule == 'cosine':
            self.beta = self.cosine_beta_schedule().to(device)
        else:
            raise ValueError(f'Unknown beta schedule {beta_schedule}')

        # all parameters
        self.alpha = 1. - self.beta 
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) 
        self.alpha_hat_prev = F.pad(self.alpha_hat[:-1],(1,0),value=1.)
        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1.-self.alpha_hat)
        self.sqrt_recip_alpha_hat = torch.sqrt(1./self.alpha_hat) # 用于估计x_0，估计x_0后用于计算p(x_{t-1}|x_t) 均值
        self.sqrt_recip_minus_alpha_hat = torch.sqrt(1./self.alpha_hat-1) 
        self.posterior_variance = (self.beta*(1.-self.alpha_hat_prev)/(1.-self.alpha_hat)) # 用于计算p(x_{t-1}|x_t)的方差
        self.posterior_mean_coef1 = (self.beta * torch.sqrt(self.alpha_hat_prev) / (1.0 - self.alphas_hat)) # 用于计算p(x_{t-1}|x_t)的均值
        self.posterior_mean_coef2 = ((1.0 - self.alphas_hat_prev)* torch.sqrt(self.alphas)/ (1.0 - self.alphas_hat))

    def _extract(self,arr,t,x_shape):
        # 根据timestep t从arr中提取对应元素并变形为x_shape
        bs = x_shape[0]
        out = arr.to(t.device).gather(0,t).float()
        out = out.reshape(bs,*((1,)*(len(x_shape)-1))) # reshape为(bs,1,1,1)
        return out
	
    def q_sample(self, x, t, noise=None):
        # q(x_t|x_0)
        if noise is None:
            Ɛ = torch.randn_like(x)
        sqrt_alpha_hat = self._extract(self.sqrt_alpha_hat,t,x.shape)
        sqrt_one_minus_alpha_hat = self._extract(self.sqrt_one_minus_alpha_hat,t,x.shape)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def q_posterior_mean_variance(self,x,x_t,t):
        # calculate mean and variance of q(x_{t-1}|x_t,x_0), we send parameters x0 and x_t into this function
        # in fact we use this function to predict p(x_{t-1}|x_t)'s mean and variance by sending x_t, \hat x_0, t
        posterior_mean =  (
            self._extract(self.posterior_mean_coef1,t,x.shape) * x 
            + self._extract(self.posterior_mean_coef2,t,x.shape) * x_t
        )
        posterior_variance = (self.posterior_variance,t,x.shape)
        return posterior_mean, posterior_variance

    def estimate_x0_from_noise(self,x_t,t,noise):
        # \hat x_0
        return (self._extract(self.sqrt_recip_alpha_hat,t,x_t.shape)*x_t + self._extract(self.sqrt_recip_minus_alpha_hat,t,x_t.shape)*noise)

    def p_mean_variance(self,model,x_t,t,clip_denoised=True):
        pred_noise = model(x_t,t)
        x_recon = self.estimate_x0_from_noise(x_t,t,pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon,min=-1.,max=1.)
        p_mean,p_var = self.q_posterior_mean_variance(x_recon,x_t,t)
        return p_mean,p_var

    def p_sample(self, model, x_t, t, clip_denoised=True):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            p_mean,p_var = self.p_mean_variance(model,x_t,t,clip_denoised=clip_denoised)
            noise = torch.randn_like(x_t)
            nonzero_mask = ((t!=0).float().view(-1,*([1]*len(x_t.shape)-1))) # 当t!=0时为1，否则为0
            pred_img = p_mean + nonzero_mask*(torch.sqrt(p_var))*noise
        return pred_img
    
    def p_sample_loop(self,model,shape):
        model.eval()
        with torch.no_grad():
            bs = shape[0]
            device = next(model.parameters()).to(device)
            img = torch.randn(shape,device=device)
            imgs = []
            for i in tqdm(reversed(range(0,self.noise_steps)),desc='sampling loop time step',total=self.noise_steps):
                img = self.p_sample(model,img,torch.full((bs,),i,device=device,dtype=torch.long)) # 从T到0
                imgs.append(img)
        return imgs
    
    @torch.no_grad()
    def sample(self,model,img_size,bs=8,channels=3):
        return self.p_sample_loop(model,(bs,channels,img_size,img_size))


def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.q_sample(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=images.shape[0])
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cuda"
    args.lr = 3e-4
    train(args)


if __name__ == '__main__':
    launch()
