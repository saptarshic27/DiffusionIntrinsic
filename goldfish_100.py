import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from models import Unet, DiffusionForwardProcess
from torch.utils.data import DataLoader, TensorDataset
from models import generate
from models import generate_batch
import  pytorch_fid_wrapper as pfw


# Assuming Unet and DiffusionForwardProcess are defined elsewhere

# --- Setup ---
OUTPUT = torch.load('data_goldfish_100.pt')
filename = f"goldfish_100.pt"
trans = transforms.Compose([transforms.Resize(28)])
OUTPUT = trans(OUTPUT)
n = 1000
img_size = 28
in_channels = 3
num_timesteps = 1000
batch_size = 128
lr = 1e-4
num_timesteps = 1000
batch_size = 128
n_epochs = 50
num_img_to_generate = 256
fid_batch_size = num_img_to_generate
n_rep = 10
num_n = 10
nn = 1
batch_size = 64   # choose based on GPU memory
num_batches = (num_img_to_generate + batch_size - 1) // batch_size
# Data Loader
# mnist_dl = DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle=True
# )

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')


# Best Loss
best_eval_loss = float('inf')

test_errors = torch.zeros((num_n, n_rep))

for rep in range(n_rep):
    for n_iter in range(num_n):
        n = (n_iter + 1)*1000
        train_data = OUTPUT[0:n].clone()    # ensure we have a tensor
        # ensure values in [0,1]
        train_data = train_data.to(torch.float32)
        train_loader = DataLoader(TensorDataset(train_data),
                          batch_size=batch_size,
                          shuffle=True, drop_last=True)
        model = Unet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        dfp = DiffusionForwardProcess()

        for epoch in range(n_epochs):
            losses = []
            model.train()
            for imgs, in tqdm(train_loader, disable=True):
                imgs = imgs.to(device)
                noise = torch.randn_like(imgs).to(device)
                t = torch.randint(0, num_timesteps,(imgs.shape[0],)).to(device)
                noisy_imgs = dfp.add_noise(imgs, noise, t)
                optimizer.zero_grad()
                noise_pred = model(noisy_imgs, t)
                loss = criterion(noise_pred, noise)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            mean_epoch_loss = np.mean(losses)
            print('Epoch:{} | Loss : {:.4f}'.format( epoch + 1, mean_epoch_loss,))
            generated_imgs = []
        # for i in tqdm(range(num_img_to_generate)):
        #     xt = generate(model, img_size, num_timesteps)
        #     xt = 255 * xt[0][0].numpy()
        #     generated_imgs.append(xt.astype(np.uint8).flatten())
        for _ in tqdm(range(num_batches)):
            curr_batch = min(batch_size, num_img_to_generate - len(generated_imgs))
            xt = generate_batch(model, curr_batch, img_size, num_timesteps)
            for i in range(curr_batch):
                img = (255 * xt[i]).numpy().astype(np.uint8)   # CHW → uint8
                generated_imgs.append(img.flatten())
        imgs_np = np.stack(generated_imgs, axis=0)       # [N,3,H,W]

        imgs_torch = torch.from_numpy(imgs_np).float()   # [0,255]
        imgs_torch = imgs_torch / 255.0                  # normalize
        imgs_torch = imgs_torch.view(-1, 3, 28, 28)


        real_slice_start = 10000
        real_slice_end = real_slice_start + fid_batch_size
        if real_slice_end > OUTPUT.shape[0]:
            # fallback: use last fid_batch_size images
            real = OUTPUT[-fid_batch_size:].cpu()
        else:
            real = OUTPUT[real_slice_start:real_slice_end].cpu()

        # compute FID (pfw.fid expects tensors and device arg as before)
        print("Generated samples:", imgs_torch.shape)
        print("Real samples:", real.shape)

        fid_score = pfw.fid(imgs_torch, real, device=device)
        test_errors[n_iter, rep] = float(fid_score)
        print(f"n_iter = {n_iter+1}, repeat = {rep+1}, FID = {fid_score:.4f}")
        torch.save(test_errors, filename)



                