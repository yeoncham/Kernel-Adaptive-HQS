import os
import json
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import ndimage

from utils import utils_image as util
from utils import utils_sisr as sr
from utils import utils_model
from utils.utils_pnp import get_rho_sigma
from models.network_unet import UNetRes as net

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Font settings for academic papers
plt.rcParams.update({
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'mathtext.fontset': 'stix',
})

def get_trajectory_path():
    possible_paths = [
        'trajectory_results/all_trajectories.json',
        'all_trajectories.json',
        '/content/all_trajectories.json',
        '/content/trajectory_results/all_trajectories.json',
        '/content/DPIR/trajectory_results/all_trajectories.json',
        '/content/drive/MyDrive/DPIR_results/all_trajectories.json'
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

def find_file(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

def plot_kernels(kernel_path, out_dir):
    print(">> Plotting Figure 2: Blur Kernels Grid...")
    try:
        kernels = np.load(kernel_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"   [Error] {kernel_path} not found.")
        return

    groups = {
        'Gaussian-small': {'indices': [0, 1, 2], 'color': '#1f77b4'},
        'Gaussian-large': {'indices': [3, 4, 5, 6, 7], 'color': '#ff7f0e'},
        'Non-Gaussian (Open-loop)': {'indices': [8, 10], 'color': '#2ca02c'},
        'Non-Gaussian (Closed-loop)': {'indices': [9, 11], 'color': '#d62728'}
    }

    fig, axes = plt.subplots(2, 6, figsize=(13, 5.5))
    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]
        kernel_img = kernels[i] if kernels.ndim == 3 else kernels.flat[i]
        ax.imshow(kernel_img, cmap='hot', interpolation='nearest')

        color = 'black'
        for g_info in groups.values():
            if i in g_info['indices']:
                color = g_info['color']
                break

        ax.set_title(f'$K_{{{i:02d}}}$', fontweight='normal', color=color, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    legend_patches = [mpatches.Patch(color=info['color'], label=name) for name, info in groups.items()]
    
    fig.legend(handles=legend_patches, 
               loc='lower center', 
               ncol=4, 
               bbox_to_anchor=(0.02, 0.02, 0.96, 0.05),
               mode="expand",
               borderaxespad=0,
               prop={'size': 11},
               frameon=True, 
               edgecolor='black', 
               fancybox=False)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.20, wspace=0.05, hspace=0.40)
    
    pdf_path = os.path.join(out_dir, 'figure2_kernels.pdf')
    png_path = os.path.join(out_dir, 'figure2_kernels.png')
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"   Saved: {pdf_path} & {png_path}")
    plt.close()

def plot_grouped_trajectories(traj_path, out_dir):
    print(">> Plotting Figure 5: Grouped Trajectories...")
    if traj_path is None or not os.path.exists(traj_path):
        print("   [Error] Trajectory JSON file not found.")
        return

    with open(traj_path, 'r') as f:
        data = json.load(f)

    groups = {
        'Gaussian-small': {'indices': [0, 1, 2], 'color': '#1f77b4'},
        'Gaussian-large': {'indices': [3, 4, 5, 6, 7], 'color': '#ff7f0e'},
        'Non-Gaussian (Open-loop)': {'indices': [8, 10], 'color': '#2ca02c'},
        'Non-Gaussian (Closed-loop)': {'indices': [9, 11], 'color': '#d62728'}
    }

    def find_data_in_json(data_list, idx):
        for item in data_list:
            if isinstance(item, dict) and item.get("kernel_index") == idx:
                return item
        return None

    def get_baseline(iterations):
        sigma_1, sigma_T, sigma_n = 49 / 255, 7.65 / 255, 7.65 / 255
        baseline_sigma = np.exp(np.log(sigma_1) + (iterations - 1) / 11 * (np.log(sigma_T) - np.log(sigma_1)))
        baseline_mu = 0.23 * (sigma_n**2) / (baseline_sigma**2)
        return baseline_sigma, baseline_mu

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    iterations = np.arange(1, 13)
    baseline_sigma, baseline_mu = get_baseline(iterations)

    ax1.plot(iterations, baseline_sigma * 255, 'k--', linewidth=1.2, label='Baseline (Hand-crafted)', zorder=5)
    ax2.plot(iterations, baseline_mu, 'k--', linewidth=1.2, label='Baseline (Hand-crafted)', zorder=5)

    found_any_data = False

    for group_name, g_info in groups.items():
        indices = g_info['indices']
        color = g_info['color']

        sigma_group, mu_group = [], []
        for idx in indices:
            traj = find_data_in_json(data, idx)
            if traj and 'sigmas' in traj and 'mus' in traj:
                sigma_group.append(np.array(traj['sigmas']) * 255)
                mu_group.append(traj['mus'])
                found_any_data = True

        if sigma_group and mu_group:
            sigma_mean, sigma_std = np.mean(sigma_group, axis=0), np.std(sigma_group, axis=0)
            mu_mean, mu_std = np.mean(mu_group, axis=0), np.std(mu_group, axis=0)

            ax1.plot(iterations, sigma_mean, label=group_name, color=color, marker='o', markersize=4.5, zorder=10)
            ax1.fill_between(iterations, sigma_mean - sigma_std, sigma_mean + sigma_std, color=color, alpha=0.15, zorder=1)

            ax2.plot(iterations, mu_mean, label=group_name, color=color, marker='s', markersize=4.5, zorder=10)
            ax2.fill_between(iterations, mu_mean - mu_std, mu_mean + mu_std, color=color, alpha=0.15, zorder=1)

    if not found_any_data:
        print("   [Error] Could not match trajectory data with kernel groups.")
        return

    ax1.set_xlabel('HQS Iteration ($k$)')
    ax1.set_ylabel(r'Denoiser Noise Level ($\sigma_k$)')
    ax1.set_ylim(0, 55)
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    ax1.set_xticks(iterations)

    ax2.set_xlabel('HQS Iteration ($k$)')
    ax2.set_ylabel(r'Penalty Parameter ($\mu_k$)')
    ax2.set_yscale('linear')
    ax2.grid(True, which="both", ls="--", alpha=0.3)
    ax2.set_xticks(iterations)

    handles, labels = ax1.get_legend_handles_labels()
    reorder_indices = [0, 3, 1, 4, 2]
    reordered_handles = [handles[i] for i in reorder_indices if i < len(handles)]
    reordered_labels = [labels[i] for i in reorder_indices if i < len(labels)]

    fig.legend(handles=reordered_handles, labels=reordered_labels, 
               loc='lower center', 
               ncol=3, 
               bbox_to_anchor=(0.06, 0.02, 0.88, 0.08), 
               mode="expand",
               borderaxespad=0,
               prop={'size': 11}, 
               frameon=True, edgecolor='black', fancybox=False)

    plt.subplots_adjust(wspace=0.22, top=0.93, bottom=0.24)
    
    pdf_path = os.path.join(out_dir, 'figure4_grouped_trajectories.pdf')
    png_path = os.path.join(out_dir, 'figure4_grouped_trajectories.png')
    plt.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"   Saved: {pdf_path} & {png_path}")
    plt.close()

def plot_individual_grids(traj_path, out_dir):
    print(">> Plotting Figure 7 & 8: Individual 4x3 Grid Trajectories...")
    if traj_path is None or not os.path.exists(traj_path):
        print("   [Error] Trajectory JSON file not found.")
        return

    with open(traj_path, 'r') as f:
        data = json.load(f)

    groups = {
        'Gaussian-small': {'indices': [0, 1, 2], 'color': '#1f77b4'},
        'Gaussian-large': {'indices': [3, 4, 5, 6, 7], 'color': '#ff7f0e'},
        'Non-Gaussian (Open-loop)': {'indices': [8, 10], 'color': '#2ca02c'},
        'Non-Gaussian (Closed-loop)': {'indices': [9, 11], 'color': '#d62728'}
    }

    def find_data_in_json(data_list, idx):
        for item in data_list:
            if isinstance(item, dict) and item.get("kernel_index") == idx:
                return item
        return None

    def get_baseline(iterations):
        sigma_1, sigma_T, sigma_n = 49 / 255, 7.65 / 255, 7.65 / 255
        baseline_sigma = np.exp(np.log(sigma_1) + (iterations - 1) / 11 * (np.log(sigma_T) - np.log(sigma_1)))
        baseline_mu = 0.23 * (sigma_n**2) / (baseline_sigma**2)
        return baseline_sigma, baseline_mu

    iterations = np.arange(1, 13)
    baseline_sigma, baseline_mu = get_baseline(iterations)

    def plot_grid(key, ylabel, filename, is_sigma=False):
        fig, axes = plt.subplots(4, 3, figsize=(12, 16.5), constrained_layout=True)
        axes = axes.flatten()

        for idx in range(12):
            traj = find_data_in_json(data, idx)
            ax = axes[idx]

            if not traj or key not in traj:
                ax.axis('off')
                continue

            color = 'black'
            for g_info in groups.values():
                if idx in g_info['indices']:
                    color = g_info['color']
                    break

            if is_sigma:
                ax.plot(iterations, baseline_sigma * 255, 'k--', linewidth=1.2)
                ax.plot(iterations, np.array(traj[key]) * 255, color=color, marker='o', markersize=4.5, linewidth=1.2)
                ax.set_ylim(0, 55)
            else:
                ax.plot(iterations, baseline_mu, 'k--', linewidth=1.2)
                ax.plot(iterations, traj[key], color=color, marker='s', markersize=4.5, linewidth=1.2)
                ax.set_yscale('linear')

            ax.set_title(f'$K_{{{idx:02d}}}$', fontweight='normal', color=color, pad=6)
            ax.grid(True, which="both", ls="--", alpha=0.3)
            ax.set_xticks([2, 4, 6, 8, 10, 12])
            ax.set_box_aspect(1.0)

            if idx % 3 == 0:
                ax.set_ylabel(ylabel)
            if idx >= 9:
                ax.set_xlabel('HQS Iteration ($k$)')

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.2)

        legend_handles = [plt.Line2D([0], [0], color='k', linestyle='--', linewidth=1.2, label='Baseline (Hand-crafted)')]
        for name, info in groups.items():
            legend_handles.append(mpatches.Patch(color=info['color'], label=name))

        final_ordered_handles = [
            legend_handles[0],  # 1행 1열: Baseline
            legend_handles[3],  # 2행 1열: Non-Gaussian (Open-loop)
            legend_handles[1],  # 1행 2열: Gaussian-small
            legend_handles[4],  # 2행 2열: Non-Gaussian (Closed-loop)
            legend_handles[2]   # 1행 3열: Gaussian-large
        ]

        fig.legend(handles=final_ordered_handles,
                   loc='lower center',
                   ncol=3,
                   bbox_to_anchor=(0.06, -0.06, 0.88, 0.04),
                   mode="expand",
                   borderaxespad=0,
                   prop={'size': 15},
                   frameon=True, edgecolor='black', fancybox=False)

        pdf_path = os.path.join(out_dir, filename + '.pdf')
        png_path = os.path.join(out_dir, filename + '.png')
        fig.savefig(pdf_path, bbox_inches='tight')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"   Saved: {pdf_path} & {png_path}")

    plot_grid('sigmas', r'Denoiser Noise Level ($\sigma_k \times 255$)', 'paper_sigma_trajectories_3x4', is_sigma=True)
    plot_grid('mus', r'Penalty Parameter ($\mu_k$)', 'paper_mu_trajectories_linear_3x4', is_sigma=False)

def plot_qualitative_comparison(img_path, kernel_path, model_ours, model_baseline, traj_path, out_dir):
    print(">> Plotting Figure 3 & 4: Qualitative Reconstruction Comparison (Separate Plots)...")
    if not os.path.exists(img_path) or not os.path.exists(kernel_path):
        print("   [Error] Required image or kernel files not found.")
        return
    if not os.path.exists(model_ours) or not os.path.exists(model_baseline):
        print("   [Error] Model weights not found in model_zoo.")
        return
    if traj_path is None or not os.path.exists(traj_path):
        print("   [Error] Trajectory JSON file not found.")
        return

    # Configuration for Simulation
    KERNEL_IDX = 8
    ITER_NUM = 12
    NOISE_LEVEL = 7.65 / 255.
    SF = 1

    # Load parameters
    with open(traj_path, 'r') as f:
        all_traj = json.load(f)
    sigmas_ours = mus_ours = None
    for t in all_traj:
        if t['kernel_index'] == KERNEL_IDX:
            sigmas_ours = np.array(t['sigmas'], dtype=np.float32)
            mus_ours = np.array(t['mus'], dtype=np.float32)
            break

    if sigmas_ours is None:
        print("   [Error] Could not read trajectory for Kernel 8.")
        return

    # Baseline schedule
    _rhos_hc, _sigmas_hc = get_rho_sigma(
        sigma=NOISE_LEVEL,
        iter_num=ITER_NUM,
        modelSigma1=49.0,
        modelSigma2=7.65,
        w=1.0
    )
    sigmas_hc = np.array(_sigmas_hc, dtype=np.float32)
    mus_hc    = np.array(_rhos_hc, dtype=np.float32)

    # Image & Kernel loading
    img_H = util.imread_uint(img_path, n_channels=3)
    img_H = util.modcrop(img_H, 8)
    
    kernels = np.load(kernel_path, allow_pickle=True)
    k = kernels[KERNEL_IDX].astype(np.float64)
    k /= (k.sum() + 1e-12)
    k_t = torch.from_numpy(np.ascontiguousarray(k[:, :, np.newaxis])).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Degradation
    img_H_f = util.uint2single(img_H)
    img_L_f = ndimage.convolve(img_H_f, np.expand_dims(k, 2), mode='wrap')
    np.random.seed(0)
    img_L_f = (img_L_f + np.random.normal(0, NOISE_LEVEL, img_L_f.shape)).astype(np.float32)
    img_L_t = util.single2tensor4(img_L_f).to(DEVICE)

    # Helper function to load model
    def load_model(path):
        model = net(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                    downsample_mode='strideconv', upsample_mode='convtranspose')
        model.load_state_dict(torch.load(path, map_location=DEVICE), strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model.to(DEVICE)

    # HQS Simulation helper
    def simulate_hqs(model, sigmas_arr, mus_arr):
        sigmas_t = torch.tensor(sigmas_arr, dtype=torch.float32, device=DEVICE)
        mus_t = torch.tensor(mus_arr, dtype=torch.float32, device=DEVICE)

        FB, FBC, F2B, FBFy = sr.pre_calculate(img_L_t, k_t, SF)
        x = img_L_t.clone()

        psnrs_x, psnrs_z, saved = [], [], {}

        with torch.no_grad():
            for i in range(ITER_NUM):
                k_step = i + 1
                tau = mus_t[i].view(1, 1, 1, 1)
                x = sr.data_solution(x, FB, FBC, F2B, FBFy, tau, SF)
                img_x_uint = util.tensor2uint(x)
                psnr_x = util.calculate_psnr(img_x_uint, img_H, border=0)
                psnrs_x.append(psnr_x)
                if k_step in [1, 4]:
                    saved[f'x{k_step}'] = (img_x_uint, psnr_x)

                sigma_map = sigmas_t[i].view(1, 1, 1, 1).expand(1, 1, x.shape[2], x.shape[3])
                x_in = torch.cat([x, sigma_map], dim=1)
                x = utils_model.test_mode(model, x_in, mode=2, refield=32, min_size=256, modulo=16)
                img_z_uint = util.tensor2uint(x)
                psnr_z = util.calculate_psnr(img_z_uint, img_H, border=0)
                psnrs_z.append(psnr_z)
                if k_step in [1, 4, 12]:
                    saved[f'z{k_step}'] = (img_z_uint, psnr_z)
        return saved, psnrs_x, psnrs_z

    # Crop/Inset coordinates
    CROP_Y, CROP_X, CROP_H, CROP_W = 95, 115, 35, 35

    def resize_nearest(img, target_w, target_h):
        h, w, c = img.shape
        y_indices = np.clip((np.arange(target_h) * (h / target_h)).astype(np.int32), 0, h - 1)
        x_indices = np.clip((np.arange(target_w) * (w / target_w)).astype(np.int32), 0, w - 1)
        return img[y_indices[:, None], x_indices]

    def draw_rectangle(img, pt1, pt2, color, thickness=1):
        h, w, c = img.shape
        x1, y1 = pt1
        x2, y2 = pt2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        color = np.array(color, dtype=img.dtype)
        img[y1:min(y1+thickness, h), x1:x2+1] = color
        img[max(y2-thickness+1, 0):y2+1, x1:x2+1] = color
        img[y1:y2+1, x1:min(x1+thickness, w)] = color
        img[y1:y2+1, max(x2-thickness+1, 0):x2+1] = color

    def add_qualitative_inset(img, bbox=(CROP_Y, CROP_X, CROP_H, CROP_W), size_pct=0.38, color_box=(0, 255, 0), color_inset=(255, 0, 0)):
        H, W, C = img.shape
        y, x, h, w = bbox
        crop = img[y:y+h, x:x+w].copy()
        inset_h, inset_w = int(H * size_pct), int(W * size_pct)
        crop_resized = resize_nearest(crop, inset_w, inset_h)
        draw_rectangle(crop_resized, (0, 0), (inset_w-1, inset_h-1), color_inset, thickness=2)
        img_out = img.copy()
        draw_rectangle(img_out, (x, y), (x+w, y+h), color_box, thickness=1)
        img_out[H-inset_h:, W-inset_w:] = crop_resized
        return img_out

    # Plot (1x6 grid)
    def run_simulation_and_plot(model_path, sigmas_arr, mus_arr, mode_name):
        print(f"   Running {mode_name.upper()} HQS simulation...")
        model = load_model(model_path)
        saved_images, psnrs_x, psnrs_z = simulate_hqs(model, sigmas_arr, mus_arr)

        fig, axes = plt.subplots(1, 6, figsize=(22, 4.5))
        plot_sequence = [
            ('x1', r'$x_1$'),
            ('z1', r'$z_1$'),
            ('x4', r'$x_4$'),
            ('z4', r'$z_4$'),
            ('z12', r'$z_{12}$')
        ]
        sub_labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
        for idx, (key, label_math) in enumerate(plot_sequence):
            ax = axes[idx]
            img_arr, psnr_val = saved_images[key]
            img_with_inset = add_qualitative_inset(img_arr)
            ax.imshow(img_with_inset)
            ax.axis('off')

        ax_curve = axes[5]
        iters_arr = np.arange(1, ITER_NUM + 1)
        ax_curve.plot(iters_arr, psnrs_x, 'o--', color='#d62728', linewidth=1.8, markersize=5.5, label=r'$x_k$')
        ax_curve.plot(iters_arr, psnrs_z, '^--', color='#1f77b4', linewidth=1.8, markersize=5.5, label=r'$z_k$')
        ax_curve.set_xticks(iters_arr)
        ax_curve.grid(True, linestyle='--', alpha=0.6)
        ax_curve.legend(loc='lower right', framealpha=0.9, edgecolor='gray')

        plt.tight_layout()
        fig.canvas.draw()
        pos_ref = axes[0].get_position()
        for i in range(5):
            ax = axes[i]
            pos = ax.get_position()
            ax.set_position([pos.x0, pos_ref.y0, pos.width, pos_ref.height])
            
        pos_curve = ax_curve.get_position()
        cur_h = pos_ref.height * 0.90
        cur_y0 = pos_ref.y0 + (pos_ref.height - cur_h)
        cur_w = cur_h * (4.5 / 22.0)
        center_x = pos_curve.x0 + pos_curve.width / 2
        cur_x0 = center_x - cur_w / 2
        ax_curve.set_position([cur_x0, cur_y0, cur_w, cur_h])
        
        for idx, (key, label_math) in enumerate(plot_sequence):
            ax = axes[idx]
            _, psnr_val = saved_images[key]
            ax.text(0.5, -0.1, f"{sub_labels[idx]} {label_math} ({psnr_val:.2f}dB)", 
                    transform=ax.transAxes, ha='center', va='center', fontsize=18)
                    
        target_y_phys = pos_ref.y0 - 0.1 * pos_ref.height
        Y_f = (target_y_phys - cur_y0) / cur_h
        ax_curve.text(0.5, Y_f, "(f) Convergence curves", transform=ax_curve.transAxes, 
                      ha='center', va='center', fontsize=18)
        
        pdf_path = os.path.join(out_dir, f'paper_qualitative_convergence_{mode_name}.pdf')
        png_path = os.path.join(out_dir, f'paper_qualitative_convergence_{mode_name}.png')
        fig.savefig(pdf_path, bbox_inches='tight')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"   Saved: {pdf_path} & {png_path}")
        plt.close(fig)

    run_simulation_and_plot(model_baseline, sigmas_hc, mus_hc, 'baseline')
    run_simulation_and_plot(model_ours, sigmas_ours, mus_ours, 'ours')

def main():
    out_dir = 'viz_results'
    os.makedirs(out_dir, exist_ok=True)

    # Paths configuration
    traj_path = get_trajectory_path()
    kernel_path = find_file(['kernels_12.npy', '/content/kernels_12.npy'])
    img_path = find_file([
        'testsets/set3c/butterfly.png',
        'butterfly.png',
        '/content/testsets/set3c/butterfly.png',
        '/content/butterfly.png'
    ])
    model_ours = find_file([
        'model_zoo/drunet_color.pth',
        'drunet_color.pth',
        '/content/model_zoo/drunet_color.pth'
    ])
    model_baseline = find_file([
        'model_zoo/drunet_color_baseline_model.pth',
        'drunet_color_baseline_model.pth',
        '/content/model_zoo/drunet_color_baseline_model.pth'
    ])

    print("==================================================================")
    print("           Unified academic plotting script for PnP-HQS            ")
    print("==================================================================")
    print(f"Device: {DEVICE}")
    print(f"Trajectory file path: {traj_path}")
    print(f"Kernels file path   : {kernel_path}")
    print(f"Test image path     : {img_path}")
    print(f"Ours model path     : {model_ours}")
    print(f"Baseline model path : {model_baseline}")
    print("------------------------------------------------------------------")

    # Call plotting functions
    plot_kernels(kernel_path, out_dir)
    plot_grouped_trajectories(traj_path, out_dir)
    plot_individual_grids(traj_path, out_dir)
    plot_qualitative_comparison(img_path, kernel_path, model_ours, model_baseline, traj_path, out_dir)
    
    print("------------------------------------------------------------------")
    print(f"All figures generated successfully inside './{out_dir}/'!")
    print("==================================================================")

if __name__ == '__main__':
    main()
