import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
import json
import time
import shutil

from utils import utils_image as util
from utils import utils_sisr as sr

def setup_colab_drive(subdir='MyDrive/DPIR_results'):
    try:
        import os
        from google.colab import drive
        if os.path.exists('/content/drive'):
            path = f'/content/drive/{subdir}'
            os.makedirs(path, exist_ok=True)
            return path
        drive.mount('/content/drive', force_remount=False)
        path = f'/content/drive/{subdir}'
        os.makedirs(path, exist_ok=True)
        return path
    except Exception as e:
        import os
        if os.path.exists('/content/drive'):
            path = f'/content/drive/{subdir}'
            os.makedirs(path, exist_ok=True)
            return path
        return None

def save_to_drive(src, drive_dir):
    if drive_dir is None:
        return
    dst = os.path.join(drive_dir, os.path.basename(src))
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH      = os.path.join('model_zoo', 'drunet_color.pth')
KERNEL_PATH     = 'kernels_12.npy'
DATA_DIR        = os.path.join('DIV2K_valid_HR', 'DIV2K_valid_HR')
RESULT_DIR      = 'trajectory_results'

N_CHANNELS      = 3
ITER_NUM        = 12
NOISE_LEVEL     = 7.65 / 255.
SF              = 1
LR              = 1e-4
SIGMA1_INIT     = 49.0
SIGMA2_INIT     = 7.65

if DEVICE.type == 'cuda':
    PATCH_SIZE      = 128
    N_IMAGES        = 100
    PATCHES_PER_IMG = 8
    BATCH_SIZE      = 16
    N_EPOCHS        = 200
else:
    PATCH_SIZE      = 128
    N_IMAGES        = 20
    PATCHES_PER_IMG = 4
    BATCH_SIZE      = 4
    N_EPOCHS        = 100
    torch.set_num_threads(os.cpu_count() or 4)

class PatchDataset(Dataset):
    def __init__(self, data_dir, n_images=100, patch_size=128, patches_per_img=4, seed=42):
        self.patch_size = patch_size
        self.patches_per_img = patches_per_img
        all_paths = sorted(util.get_image_paths(data_dir))
        self.img_paths = all_paths[:min(n_images, len(all_paths))]
        self.rng = np.random.RandomState(seed)
        
        self.cached_images = []
        for path in self.img_paths:
            img_H = util.imread_uint(path, n_channels=N_CHANNELS)
            img_H = util.modcrop(img_H, 8)
            self.cached_images.append(img_H)

    def __len__(self):
        return len(self.cached_images) * self.patches_per_img

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_img
        img_H = self.cached_images[img_idx]
        H, W, _ = img_H.shape
        ps = self.patch_size
        
        if H < ps or W < ps:
            pad_h = max(0, ps - H)
            pad_w = max(0, ps - W)
            img_H = np.pad(img_H, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            H, W, _ = img_H.shape
            
        rnd_h = self.rng.randint(0, H - ps + 1)
        rnd_w = self.rng.randint(0, W - ps + 1)
        patch = img_H[rnd_h:rnd_h+ps, rnd_w:rnd_w+ps, :]
        
        patch_t = torch.from_numpy(
            np.ascontiguousarray(patch.transpose(2, 0, 1))
        ).float() / 255.0
        return patch_t

def load_frozen_drunet(model_path, device):
    from models.network_unet import UNetRes as net
    model = net(
        in_nc=N_CHANNELS + 1,
        out_nc=N_CHANNELS,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode='R',
        downsample_mode='strideconv',
        upsample_mode='convtranspose'
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    return model

def convolve_circular(img, k):
    kh, kw = k.shape[-2:]
    pad_h = kh // 2
    pad_w = kw // 2
    img_padded = torch.nn.functional.pad(img, (pad_w, pad_w, pad_h, pad_h), mode='circular')
    k_flipped = torch.flip(k, dims=(-2, -1))
    C = img.shape[1]
    k_expanded = k_flipped.expand(C, 1, kh, kw)
    blurred = torch.nn.functional.conv2d(img_padded, k_expanded, groups=C)
    return blurred

def hqs_forward(img_L_tensor, k_tensor, model, log_sigmas, log_mus):
    sigmas = torch.exp(log_sigmas)
    mus    = torch.exp(log_mus)
    FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_tensor, k_tensor, SF)
    x = img_L_tensor.clone()

    for i in range(len(sigmas)):
        tau = mus[i].view(1, 1, 1, 1)
        x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, SF)
        sigma_map = sigmas[i].view(1, 1, 1, 1).expand(x.shape[0], 1, x.shape[2], x.shape[3])
        x_input = torch.cat([x, sigma_map], dim=1)
        x = model(x_input)
    return x

def train_kernel(k_index, kernel_np, model, dataloader, device, result_dir):
    sigma_init = np.logspace(
        np.log10(SIGMA1_INIT), np.log10(SIGMA2_INIT), ITER_NUM
    ).astype(np.float32) / 255.0

    mu_init = np.array([
        0.23 * (NOISE_LEVEL ** 2) / (s ** 2) for s in sigma_init
    ], dtype=np.float32)
    mu_init = np.clip(mu_init, 1e-5, 20.0)

    log_sigmas = nn.Parameter(torch.log(torch.tensor(sigma_init, device=device)))
    log_mus    = nn.Parameter(torch.log(torch.tensor(mu_init,    device=device)))

    optimizer = optim.Adam([log_sigmas, log_mus], lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=LR * 0.01)

    k = kernel_np.astype(np.float64)
    k = k / (k.sum() + 1e-12)

    k_t = torch.from_numpy(
        np.ascontiguousarray(k[:, :, np.newaxis])
    ).float()
    k_t = k_t.permute(2, 0, 1).unsqueeze(0).to(device)

    best_loss = float('inf')
    best_log_sigmas = log_sigmas.data.clone()
    best_log_mus    = log_mus.data.clone()

    history = {'loss': [], 'psnr': []}
    t_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        epoch_loss = 0.0
        n_steps    = 0

        for img_H_batch in dataloader:
            img_H_t = img_H_batch.to(device)
            img_L_t = convolve_circular(img_H_t, k_t)
            noise = torch.randn_like(img_L_t) * NOISE_LEVEL
            img_L_t = img_L_t + noise

            x_out = hqs_forward(img_L_t, k_t, model, log_sigmas, log_mus)
            loss = nn.functional.mse_loss(x_out, img_H_t)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_([log_sigmas, log_mus], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                log_sigmas.clamp_(math.log(0.5 / 255.), math.log(50. / 255.))
                log_mus.clamp_(math.log(1e-6), math.log(100.0))

            epoch_loss += loss.item()
            n_steps    += 1

        scheduler.step()
        avg_loss = epoch_loss / n_steps
        avg_psnr = -10 * math.log10(avg_loss + 1e-10)
        history['loss'].append(avg_loss)
        history['psnr'].append(avg_psnr)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_log_sigmas = log_sigmas.data.clone()
            best_log_mus    = log_mus.data.clone()

    best_sigmas = torch.exp(best_log_sigmas).cpu().numpy()
    best_mus    = torch.exp(best_log_mus).cpu().numpy()

    result = {
        'kernel_index':   k_index,
        'sigmas':         best_sigmas.tolist(),
        'sigmas_x255':    (best_sigmas * 255).tolist(),
        'mus':            best_mus.tolist(),
        'best_psnr_dB':   -10 * math.log10(best_loss + 1e-10),
        'history_loss':   history['loss'],
        'history_psnr':   history['psnr'],
    }

    json_path = os.path.join(result_dir, f'kernel_{k_index:02d}_trajectory.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    np.save(os.path.join(result_dir, f'kernel_{k_index:02d}_sigmas.npy'), best_sigmas)
    np.save(os.path.join(result_dir, f'kernel_{k_index:02d}_mus.npy'),    best_mus)
    return result

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    drive_dir = setup_colab_drive()

    kernels = np.load(KERNEL_PATH, allow_pickle=True)

    if kernels.dtype == object:
        kernel_list = [np.array(k, dtype=np.float64) for k in kernels.flat]
    elif kernels.ndim == 3:
        kernel_list = [kernels[i].astype(np.float64) for i in range(kernels.shape[0])]
    elif kernels.ndim == 2:
        kernel_list = [kernels.astype(np.float64)]
    else:
        raise ValueError(f'Unsupported kernel shape: {kernels.shape}')

    dataset = PatchDataset(
        DATA_DIR,
        n_images=N_IMAGES,
        patch_size=PATCH_SIZE,
        patches_per_img=PATCHES_PER_IMG
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    model = load_frozen_drunet(MODEL_PATH, DEVICE)

    all_results = []
    for k_idx, k_np in enumerate(kernel_list):
        result = train_kernel(k_idx, k_np, model, dataloader, DEVICE, RESULT_DIR)
        all_results.append(result)
        save_to_drive(RESULT_DIR, drive_dir)

    summary_path = os.path.join(RESULT_DIR, 'all_trajectories.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    save_to_drive(RESULT_DIR, drive_dir)

if __name__ == '__main__':
    main()