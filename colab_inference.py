"""
Championship SR - Google Colab Inference Script
=================================================
Run 4× super-resolution on DIV2K test images using the
CompleteEnhancedFusionSR model (Phase 5 checkpoint).

Usage on Colab:
    1. Upload this script and the project repo to your Drive
    2. Mount Google Drive
    3. Run all cells

Author: NTIRE SR Team
"""

# ============================================================================
# Cell 1: Setup & Installation
# ============================================================================
# Uncomment these lines when running in Google Colab:
#
# from google.colab import drive
# drive.mount('/content/drive')
#
# # Clone the project repo
# !git clone https://github.com/Nikhil-AI-Labs/image-super-resolution-2.git /content/project
#
# # Install dependencies (torch & torchvision are pre-installed on Colab)
# !pip install tqdm pyyaml pillow numpy
#
# # Download DIV2K test LR images (if not already on Drive)
# # !wget https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_test_LR_bicubic_X4.zip -O /content/DIV2K_test.zip
# # !unzip -q /content/DIV2K_test.zip -d /content/project/

import os
import sys
import time
import glob
import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ============================================================================
# Cell 2: Configuration
# ============================================================================

# ==================== EDIT THESE PATHS ====================
# Path to the project root (contains src/, configs/, etc.)
# For local: use your local project path
# For Colab: use "/content/project" (after cloning the repo)
PROJECT_ROOT = "/content/project"  # <-- COLAB DEFAULT
# PROJECT_ROOT = r"d:\image super resolution"  # <-- LOCAL WINDOWS

# Path to the trained checkpoint (included in the GitHub repo)
CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "checkpoints", "phase5_single_gpu",
    "championship_sr_phase5_single_gpu",
    "best_epoch0050_psnr30.05.pth"
)

# Path to the test LR images directory
TEST_LR_DIR = os.path.join(PROJECT_ROOT, "DIV2K_test_LR_bicubic", "X4")

# Output directory for super-resolved images
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "colab_inference")

# Path to pretrained expert weights (HAT, DAT, NAFNet)
# These are too large for GitHub (~80MB each), so load from Google Drive.
# Upload the pretrained/ folder to your Drive and set this path:
PRETRAINED_DIR = "/content/drive/MyDrive/pretrained"  # <-- COLAB DEFAULT
# PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "pretrained")  # <-- LOCAL

# ==================== MODEL CONFIG ====================
# These MUST match the settings used during training (from train_config.yaml)
MODEL_CONFIG = {
    "scale": 4,
    "num_experts": 3,
    "fusion_dim": 64,
    "num_heads": 4,
    "refine_depth": 4,
    "refine_channels": 64,
    "num_bands": 3,
    "block_size": 8,
    # All improvements enabled (as per Phase 5 training)
    "enable_hierarchical": True,
    "enable_multi_domain_freq": True,
    "enable_lka": True,
    "enable_edge_enhance": True,
    "enable_dynamic_selection": True,
    "enable_cross_band_attn": True,
    "enable_adaptive_bands": True,
    "enable_multi_resolution": True,
    "enable_collaborative": True,
}

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NOTE: FP16 is DISABLED because HAT does not support half precision.
# All inference runs in FP32.
USE_FP16 = False

# ============================================================================
# Cell 3: Add project to path & import model
# ============================================================================

# Add the project root to the Python path so we can import from src/
sys.path.insert(0, PROJECT_ROOT)

from src.models.enhanced_fusion import CompleteEnhancedFusionSR
from src.models import expert_loader


def build_expert_ensemble(device: str = "cuda") -> expert_loader.ExpertEnsemble:
    """
    Build the ExpertEnsemble with the 3 frozen experts: HAT, DAT, NAFNet.
    
    The expert ensemble is needed to construct the full model architecture
    so that checkpoint loading works (state_dict keys must match).
    """
    print("=" * 60)
    print("LOADING EXPERT MODELS")
    print("=" * 60)
    
    ensemble = expert_loader.ExpertEnsemble(
        scale=MODEL_CONFIG["scale"],
        device=device,
    )
    
    # Expert checkpoint paths
    expert_paths = {
        "hat": os.path.join(PRETRAINED_DIR, "hat", "HAT-L_SRx4_ImageNet-pretrain.pth"),
        "dat": os.path.join(PRETRAINED_DIR, "dat", "DAT_x4.pth"),
        "nafnet": os.path.join(PRETRAINED_DIR, "nafnet", "NAFNet-SIDD-width64.pth"),
    }
    
    # Verify expert weight files exist
    for name, path in expert_paths.items():
        if os.path.exists(path):
            print(f"  ✓ Found {name} weights: {os.path.basename(path)}")
        else:
            print(f"  ⚠ Missing {name} weights: {path}")
            print(f"    Expert will be initialized with random weights.")
            print(f"    This is OK — trained checkpoint will overwrite them.")
    
    # Load experts (they'll be frozen during inference anyway)
    results = ensemble.load_all_experts(
        checkpoint_paths=expert_paths,
        freeze=True,
    )
    
    for name, success in results.items():
        status = "✓ loaded" if success else "⚠ random init"
        print(f"  {name}: {status}")
    
    return ensemble


def build_model(device: str = "cuda") -> CompleteEnhancedFusionSR:
    """
    Build the CompleteEnhancedFusionSR model matching the training config.
    
    Returns the model with random weights — checkpoint loading happens next.
    """
    print("\n" + "=" * 60)
    print("BUILDING MODEL")
    print("=" * 60)
    
    # Step 1: Build expert ensemble
    ensemble = build_expert_ensemble(device)
    
    # Step 2: Build the fusion model with matching architecture params
    model = CompleteEnhancedFusionSR(
        expert_ensemble=ensemble,
        num_experts=MODEL_CONFIG["num_experts"],
        num_bands=MODEL_CONFIG["num_bands"],
        block_size=MODEL_CONFIG["block_size"],
        upscale=MODEL_CONFIG["scale"],
        fusion_dim=MODEL_CONFIG["fusion_dim"],
        num_heads=MODEL_CONFIG["num_heads"],
        refine_depth=MODEL_CONFIG["refine_depth"],
        refine_channels=MODEL_CONFIG["refine_channels"],
        enable_hierarchical=MODEL_CONFIG["enable_hierarchical"],
        enable_multi_domain_freq=MODEL_CONFIG["enable_multi_domain_freq"],
        enable_lka=MODEL_CONFIG["enable_lka"],
        enable_edge_enhance=MODEL_CONFIG["enable_edge_enhance"],
        enable_dynamic_selection=MODEL_CONFIG["enable_dynamic_selection"],
        enable_cross_band_attn=MODEL_CONFIG["enable_cross_band_attn"],
        enable_adaptive_bands=MODEL_CONFIG["enable_adaptive_bands"],
        enable_multi_resolution=MODEL_CONFIG["enable_multi_resolution"],
        enable_collaborative=MODEL_CONFIG["enable_collaborative"],
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n  Architecture: CompleteEnhancedFusionSR")
    print(f"  Total parameters:     {total_params:>12,}")
    print(f"  Trainable parameters: {trainable_params:>12,}")
    print(f"  Frozen parameters:    {frozen_params:>12,}")
    print(f"  Device:               {device}")
    
    return model


def load_checkpoint(model: CompleteEnhancedFusionSR, checkpoint_path: str, device: str = "cuda"):
    """
    Load the trained checkpoint into the model.
    
    The checkpoint contains 'model_state_dict' with weights for the entire
    model (experts + fusion components).
    """
    print("\n" + "=" * 60)
    print("LOADING CHECKPOINT")
    print("=" * 60)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please upload the checkpoint to the correct path."
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Show checkpoint info
    epoch = checkpoint.get("epoch", "unknown")
    metrics = checkpoint.get("metrics", {})
    timestamp = checkpoint.get("timestamp", "unknown")
    
    print(f"  Epoch:     {epoch}")
    print(f"  Timestamp: {timestamp}")
    if metrics:
        for k, v in metrics.items():
            print(f"  {k}:      {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Load model weights
    state_dict = checkpoint["model_state_dict"]
    
    # Try strict loading first, fall back to non-strict
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"\n  ✓ Loaded all {len(state_dict)} weight tensors (strict=True)")
    except RuntimeError as e:
        print(f"\n  ⚠ Strict loading failed, trying non-strict...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Loaded weights (strict=False)")
        if missing:
            print(f"  ⚠ Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"      - {k}")
            if len(missing) > 5:
                print(f"      ... and {len(missing) - 5} more")
        if unexpected:
            print(f"  ⚠ Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"      - {k}")
            if len(unexpected) > 5:
                print(f"      ... and {len(unexpected) - 5} more")
    
    return checkpoint


# ============================================================================
# Cell 4: Image Loading Utilities
# ============================================================================

def load_image(path: str) -> torch.Tensor:
    """
    Load an image and convert to a float32 tensor in [0, 1] range.
    
    Args:
        path: Path to the image file
        
    Returns:
        Tensor of shape [1, 3, H, W] in [0, 1] range
    """
    img = Image.open(path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0  # [H, W, 3], [0, 1]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return img_tensor


def save_image(tensor: torch.Tensor, path: str):
    """
    Save a tensor as a PNG image.
    
    Args:
        tensor: Image tensor of shape [1, 3, H, W] or [3, H, W] in [0, 1] range
        path: Output file path
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Clamp to [0, 1] and convert to uint8
    tensor = tensor.clamp(0, 1)
    img_np = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    
    img = Image.fromarray(img_np)
    img.save(path, format="PNG")


def get_test_images(test_dir: str) -> list:
    """
    Get sorted list of test image paths from the LR directory.
    
    Args:
        test_dir: Path to the test LR images directory
        
    Returns:
        Sorted list of image file paths
    """
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(test_dir, ext)))
    
    image_paths.sort()
    return image_paths


# ============================================================================
# Cell 5: Inference
# ============================================================================

@torch.no_grad()
def run_inference(
    model: CompleteEnhancedFusionSR,
    test_dir: str,
    output_dir: str,
    device: str = "cuda",
    use_fp16: bool = True,
):
    """
    Run inference on all test images.
    
    Args:
        model: The loaded CompleteEnhancedFusionSR model
        test_dir: Path to directory containing LR test images
        output_dir: Path to save SR output images
        device: Device to run inference on
        use_fp16: Whether to use half precision (faster on GPU)
    """
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test images
    image_paths = get_test_images(test_dir)
    
    if not image_paths:
        print(f"  ⚠ No images found in: {test_dir}")
        print(f"    Expected .png files in the directory.")
        return
    
    print(f"  Found {len(image_paths)} test images")
    print(f"  Output directory: {output_dir}")
    print(f"  Device: {device}")
    print(f"  Using FP16: {use_fp16 and device == 'cuda'}")
    print()
    
    model.eval()
    
    total_time = 0
    results = []
    
    for i, img_path in enumerate(tqdm(image_paths, desc="Super-resolving")):
        img_name = os.path.basename(img_path)
        output_name = img_name.replace("x4", "_SR")
        if output_name == img_name:
            # If no 'x4' in name, just add _SR prefix
            name, ext = os.path.splitext(img_name)
            output_name = f"{name}_SR{ext}"
        output_path = os.path.join(output_dir, output_name)
        
        # Load image (always FP32 — HAT doesn't support FP16)
        lr_img = load_image(img_path).to(device)
        
        # Run model
        start_time = time.time()
        
        try:
            sr_img = model(lr_img)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n  ⚠ OOM on {img_name}. Trying with float32...")
                torch.cuda.empty_cache()
                sr_img = model(lr_img)
            else:
                raise
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        
        # Save output
        save_image(sr_img, output_path)
        
        # Store result info
        lr_h, lr_w = lr_img.shape[2], lr_img.shape[3]
        sr_h, sr_w = sr_img.shape[2], sr_img.shape[3]
        results.append({
            "name": img_name,
            "lr_size": f"{lr_w}×{lr_h}",
            "sr_size": f"{sr_w}×{sr_h}",
            "time": elapsed,
        })
        
        # Print progress for first few and every 10th
        if i < 3 or (i + 1) % 20 == 0:
            tqdm.write(
                f"  {img_name}: {lr_w}×{lr_h} → {sr_w}×{sr_h} "
                f"({elapsed:.2f}s)"
            )
    
    # Summary
    avg_time = total_time / len(image_paths) if image_paths else 0
    print(f"\n" + "=" * 60)
    print(f"INFERENCE COMPLETE")
    print(f"=" * 60)
    print(f"  Images processed: {len(image_paths)}")
    print(f"  Total time:       {total_time:.1f}s")
    print(f"  Average time:     {avg_time:.2f}s per image")
    print(f"  Output saved to:  {output_dir}")
    
    return results


# ============================================================================
# Cell 6: Visualization (Optional)
# ============================================================================

def visualize_results(
    test_dir: str,
    output_dir: str,
    num_samples: int = 4,
    save_path: str = None,
):
    """
    Display side-by-side LR/SR comparisons.
    
    Args:
        test_dir: Path to LR test images
        output_dir: Path to SR output images
        num_samples: Number of images to display
        save_path: Optional path to save the comparison figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization.")
        return
    
    lr_paths = sorted(get_test_images(test_dir))
    sr_paths = sorted(glob.glob(os.path.join(output_dir, "*.png")))
    
    if not sr_paths:
        print("No SR images found to visualize.")
        return
    
    num_samples = min(num_samples, len(sr_paths))
    
    # Select evenly spaced samples
    indices = np.linspace(0, len(sr_paths) - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(indices):
        lr_img = Image.open(lr_paths[sample_idx]).convert("RGB")
        sr_img = Image.open(sr_paths[sample_idx]).convert("RGB")
        
        lr_name = os.path.basename(lr_paths[sample_idx])
        sr_name = os.path.basename(sr_paths[sample_idx])
        
        axes[idx, 0].imshow(lr_img)
        axes[idx, 0].set_title(f"LR Input ({lr_img.size[0]}×{lr_img.size[1]})\n{lr_name}", fontsize=11)
        axes[idx, 0].axis("off")
        
        axes[idx, 1].imshow(sr_img)
        axes[idx, 1].set_title(f"SR Output ({sr_img.size[0]}×{sr_img.size[1]})\n{sr_name}", fontsize=11)
        axes[idx, 1].axis("off")
    
    plt.suptitle("Championship SR — 4× Super Resolution Results", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Comparison saved to: {save_path}")
    
    plt.show()
    plt.close()


# ============================================================================
# Cell 7: Main Execution
# ============================================================================

def main():
    """Main entry point for the inference pipeline."""
    
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Championship SR — DIV2K Test Inference Pipeline     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Device:     {DEVICE}")
    print(f"  Checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
    print(f"  Test dir:   {TEST_LR_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print()
    
    # ---- Step 1: Build model ----
    model = build_model(device=DEVICE)
    
    # ---- Step 2: Load checkpoint ----
    load_checkpoint(model, CHECKPOINT_PATH, device=DEVICE)
    
    # ---- Step 3: Run inference ----
    results = run_inference(
        model=model,
        test_dir=TEST_LR_DIR,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        use_fp16=USE_FP16,
    )
    
    # ---- Step 4: Visualize (optional) ----
    comparison_path = os.path.join(OUTPUT_DIR, "comparison.png")
    visualize_results(
        test_dir=TEST_LR_DIR,
        output_dir=OUTPUT_DIR,
        num_samples=4,
        save_path=comparison_path,
    )
    
    print("\n✓ All done!")


if __name__ == "__main__":
    main()
