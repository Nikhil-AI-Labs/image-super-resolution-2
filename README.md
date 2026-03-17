# [NTIRE 2026 Challenge on Image Super-Resolution (x4)](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

## Team 29 — FreqFusion: Multi-Expert Fusion via Frequency-Guided Hierarchical Attention Networks

**Team Name:** Anant_SVNIT  
**Team ID:** TEAM_29  
**Team Leader:** Nikhil Pathak (SVNIT, Surat, India)  
**Affiliations:** Sardar Vallabhbhai National Institute of Technology (SVNIT) · Norwegian University of Science and Technology (NTNU)

---

## Notice
All submitted code must follow the format defined in this repository. Submissions that do not follow the required format may be rejected during the final evaluation stage.

After the challenge ends, we will release all submitted code as open-source for reproducibility. If you would like your model to remain confidential, please contact the organizers in advance.

## How to test the baseline model?

1. Clone the repository:
```bash
git clone https://github.com/Nikhil-AI-Labs/image-super-resolution-2.git
cd image-super-resolution-2
```

2. Install dependencies:
```bash
conda create -n NTIRE-SR python=3.8
conda activate NTIRE-SR
pip install -r requirements.txt
```

3. Download pretrained expert weights from Google Drive:

📥 **[Download Expert Weights from Google Drive](https://drive.google.com/drive/folders/1m8cMpiqlAzOT2-2S2x0OhMSu4Fcfqj-A?usp=sharing)**

Place the downloaded weights in the `pretrained/` directory with the following structure:
```
pretrained/
├── hat/
│   └── HAT-L_SRx4_ImageNet-pretrain.pth
├── dat/
│   └── DAT_x4.pth
└── nafnet/
    └── NAFNet-SIDD-width64.pth
```

4. The fusion model checkpoint is already included in the repository:
```
checkpoints/phase5_single_gpu/championship_sr_phase5_single_gpu/best_epoch0050_psnr30.05.pth
```

5. Run inference:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 29
```

- You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure to change the directories `--valid_dir`/`--test_dir` and `--save_dir`.
- Our model (team29): FreqFusion (default, `--model_id 29`).

### Quick Example

```bash
# Run on validation set
CUDA_VISIBLE_DEVICES=0 python test.py \
    --valid_dir ./DIV2K_test_LR_bicubic/X4 \
    --save_dir ./results \
    --model_id 29

# Run on test set
CUDA_VISIBLE_DEVICES=0 python test.py \
    --test_dir /path/to/NTIRE2026_test_LR \
    --save_dir ./results \
    --model_id 29
```

The output SR images will be saved in `./results/29_FreqFusion_team29/valid/` or `./results/29_FreqFusion_team29/test/`.

## How to add your model to this baseline?

> [!IMPORTANT]
> 🚨 Submissions that do not follow the official format will be rejected.

1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1sEliBQf27EEN2bzQUO-XZaTdVG8SYWNouKSHqRYY9mE/edit?usp=sharing) and get your team ID.

2. Put your the code of your model in folder: `./models/[Your_Team_ID]_[Your_Model_Name]`
   - Please zero pad `[Your_Team_ID]` into two digits: e.g. 00, 01, 02

3. Put the pretrained model in folder: `./model_zoo/[Your_Team_ID]_[Your_Model_Name]`
   - Please zero pad `[Your_Team_ID]` into two digits: e.g. 00, 01, 02
   - Note: Please provide a download link for the pretrained model, if the file size exceeds 100 MB. Put the link in `./model_zoo/[Your_Team_ID]_[Your_Model_Name]/[Your_Team_ID]_[Your_Model_Name].txt`

4. Add your model to the model loader [test.py](test.py) as follows:
   - Edit the `else` to `elif` in [test.py](test.py), and then you can add your own model with model id.
   - `model_func` must be a function, which accept 4 params:
     - `model_dir`: the pretrained model path
     - `input_path`: a folder containing several images in PNG format
     - `output_path`: a folder for restored images in PNG format
     - `device`: computation device

5. Send us the command to download your code, e.g,
```bash
git clone https://github.com/Nikhil-AI-Labs/image-super-resolution-2.git
```

> [!TIP]
> Your model code does not need to be fully refactored to fit this repository. Instead, you may add a lightweight external interface (e.g., `models/team29_FreqFusion/io.py`) that wraps your existing code, while keeping the original implementation unchanged.

## Our Model: FreqFusion (Team 29)

### Architecture Overview

FreqFusion is a 7-phase multi-expert fusion architecture for ×4 single-image super-resolution. Instead of training a single massive SR model from scratch, we leverage three powerful pre-trained (frozen) expert SR models and train a lightweight fusion network (~1.2M trainable parameters) that intelligently combines their outputs.

```
LR Input [B, 3, H, W]
       │
       ▼
┌──────────────────────────────────┐
│  Phase 1: Frozen Experts (~100M) │
│  ┌────────┐ ┌──────┐ ┌────────┐ │
│  │ HAT-L  │ │ DAT  │ │NAFNet64│ │
│  │ 40.84M │ │11.21M│ │  67M   │ │
│  └────┬───┘ └──┬───┘ └───┬────┘ │
└───────┼────────┼─────────┼──────┘
        └────────┴─────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
  Phase 2:  Phase 3:  Phase 4:
  Multi-Domain Cross-Band Collaborative
  Freq Decomp  Attn+LKA  Feature Learning
  (DCT+DWT+FFT) (k=21)   + LKA
        │        │        │
        └────────┴────────┘
                 │
                 ▼
         Phase 5: Hierarchical
         Multi-Res Fusion (64→128→256)
                 │
                 ▼
         Phase 6: Dynamic Expert
         Selection (per-pixel gating)
                 │
                 ▼
         Phase 7: Refinement +
         Edge Enhancement
                 │
                 ▼
         SR Output [B, 3, 4H, 4W]
```

### Expert Models (Frozen)

| Expert | Params | Feat. Dim | Architecture |
|--------|--------|-----------|-------------|
| HAT-L | 40.84M | 180 | Hybrid Attention Transformer |
| DAT | 11.21M | 180 | Dual Aggregation Transformer |
| NAFNet-64 | ~67M | 64 | Nonlinear Act.-Free Net |

### Key Innovations

1. **Multi-domain 9-band frequency decomposition** combining DCT + DWT (db4) + FFT with learnable adaptive masks
2. **Decomposed Large Kernel Attention (k=21)** for global spatial context with O(k) parameter complexity
3. **Difficulty-adaptive per-pixel expert gating** with learnable temperature
4. **Cached training pipeline** achieving 10–20× speedup

### Pretrained Model Weights

| Component | Size | Location |
|-----------|------|----------|
| **Fusion Checkpoint** | ~12 MB | Included in repo: `checkpoints/phase5_single_gpu/championship_sr_phase5_single_gpu/best_epoch0050_psnr30.05.pth` |
| **Expert Weights** (HAT-L, DAT, NAFNet) | ~170 MB total | [Google Drive](https://drive.google.com/drive/folders/1m8cMpiqlAzOT2-2S2x0OhMSu4Fcfqj-A?usp=sharing) |

### Training

- **Optimizer:** AdamW (β₁=0.9, β₂=0.999, wd=10⁻⁴)
- **Learning Rate:** 2×10⁻⁴ with CosineAnnealingWarmRestarts
- **Batch Size:** 8 (4× gradient accumulation = effective 32)
- **Patch Size:** 64×64 LR
- **Total Epochs:** 150
- **EMA Decay:** 0.9995
- **Training Data:** DF2K (DIV2K + Flickr2K)
- **Trainable Parameters:** ~1.2M (frozen experts: ~100M)

### Repository Structure

```
.
├── factsheet/                      # NTIRE 2026 Fact Sheet (LaTeX)
├── model_zoo/
│   └── team29_FreqFusion/          # Pretrained model download links
│       └── team29_FreqFusion.txt
├── models/
│   └── team29_FreqFusion/          # Model interface (io.py wrapper)
│       ├── __init__.py
│       └── io.py
├── utils/                          # Official NTIRE utils
│   ├── model_summary.py
│   ├── utils_image.py
│   └── utils_logger.py
├── src/                            # Full model implementation
│   ├── data/                       # Dataset & augmentation
│   ├── losses/                     # Loss functions (L1, SWT, FFT, SSIM)
│   ├── models/                     # Model architectures
│   │   ├── enhanced_fusion.py      # CompleteEnhancedFusionSR
│   │   ├── expert_loader.py        # Expert ensemble loader
│   │   ├── fusion_network.py       # Core fusion components
│   │   ├── multi_domain_frequency.py
│   │   ├── large_kernel_attention.py
│   │   ├── edge_enhancement.py
│   │   ├── hierarchical_fusion.py
│   │   ├── dat/                    # DAT architecture
│   │   ├── hat/                    # HAT architecture
│   │   └── nafnet/                 # NAFNet architecture
│   ├── training/                   # Training utilities
│   └── utils/                      # Metrics & checkpointing
├── configs/
│   └── train_config.yaml           # Training configuration
├── scripts/                        # Utility scripts
├── checkpoints/                    # Trained fusion checkpoint (included)
├── pretrained/                     # Expert weights (download from Drive)
├── LICENSE                         # MIT License
├── README.md                       # This file
├── eval.py                         # IQA evaluation script
├── requirements.txt                # Python dependencies
├── test.py                         # Official NTIRE test runner
├── train.py                        # Training script
└── colab_inference.py              # Google Colab inference script
```

## How to eval images using IQA metrics?

### Environments
```bash
conda create -n NTIRE-SR python=3.8
conda activate NTIRE-SR
pip install -r requirements.txt
```

### Folder Structure
```
test_dir
├── HR
│   ├── 0901.png
│   ├── 0902.png
│   ├── ...
├── LQ
│   ├── 0901x4.png
│   ├── 0902x4.png
│   ├── ...
    
output_dir
├── 0901x4.png
├── 0902x4.png
├──...
```

### Command to calculate metrics
```bash
python eval.py \
--output_folder "/path/to/your/output_dir" \
--target_folder "/path/to/test_dir/HR" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
```

The `eval.py` file accepts the following 4 parameters:
- `output_folder`: Path where the restored images are saved.
- `target_folder`: Path to the HR images in the test dataset. This is used to calculate FR-IQA metrics.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `device`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.

### Weighted score for Perception Quality Track
We use the following equation to calculate the final weight score:

$$\text{Score} = \left(1 - \text{LPIPS}\right) + \left(1 - \text{DISTS}\right) + \text{CLIPIQA} + \text{MANIQA} + \frac{\text{MUSIQ}}{100} + \max\left(0, \frac{10 - \text{NIQE}}{10}\right).$$

The score is calculated on the averaged IQA scores.

## NTIRE Image SR ×4 Challenge Series
Code repositories and accompanying technical report PDFs for each edition:
- NTIRE 2025: [CODE](https://github.com/zhengchen1999/NTIRE2025_ImageSR_x4) | [PDF](https://arxiv.org/pdf/2504.14582)
- NTIRE 2024: [CODE](https://github.com/zhengchen1999/NTIRE2024_ImageSR_x4) | [PDF](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Chen_NTIRE_2024_Challenge_on_Image_Super-Resolution_x4_Methods_and_Results_CVPRW_2024_paper.pdf)
- NTIRE 2023: [CODE](https://github.com/zhengchen1999/NTIRE2023_ImageSR_x4) | [PDF](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/papers/Zhang_NTIRE_2023_Challenge_on_Image_Super-Resolution_x4_Methods_and_Results_CVPRW_2023_paper.pdf)

## License and Acknowledgement
This code repository is released under [MIT License](LICENSE).
