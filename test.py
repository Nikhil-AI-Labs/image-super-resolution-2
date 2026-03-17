import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from utils import utils_logger
from utils import utils_image as util


def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    model_id = args.model_id
    if model_id == 29:
        # Team 29: FreqFusion — Multi-Expert Fusion via Frequency-Guided
        # Hierarchical Attention Networks
        from models.team29_FreqFusion import main as FreqFusion
        name = f"{model_id:02}_FreqFusion_team29"
        model_path = os.path.join(
            'checkpoints', 'phase5_single_gpu',
            'championship_sr_phase5_single_gpu',
            'best_epoch0050_psnr30.05.pth'
        )
        model_func = FreqFusion
    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    return model_func, model_path, name


def run(model_func, model_name, model_path, device, args, mode="test"):
    # --------------------------------
    # dataset path
    # --------------------------------
    if mode == "valid":
        data_path = args.valid_dir
    elif mode == "test":
        data_path = args.test_dir
    assert data_path is not None, "Please specify the dataset path for validation or test."

    save_path = os.path.join(args.save_dir, model_name, mode)
    util.mkdir(save_path)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    model_func(model_dir=model_path, input_path=data_path, output_path=save_path, device=device)
    end.record()
    torch.cuda.synchronize()
    print(f"Model {model_name} runtime (Including I/O): {start.elapsed_time(end)} ms")


def main(args):
    utils_logger.logger_info("NTIRE2026-ImageSRx4", log_path="NTIRE2026-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2026-ImageSRx4")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    model_func, model_path, model_name = select_model(args, device)
    logger.info(model_name)

    # if model not in results:
    if args.valid_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="valid")

    if args.test_dir is not None:
        run(model_func, model_name, model_path, device, args, mode="test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2026-ImageSRx4")
    parser.add_argument("--valid_dir", default=None, type=str, help="Path to the validation set")
    parser.add_argument("--test_dir", default=None, type=str, help="Path to the test set")
    parser.add_argument("--save_dir", default="NTIRE2026-ImageSRx4/results", type=str)
    parser.add_argument("--model_id", default=29, type=int)

    args = parser.parse_args()
    pprint(args)

    main(args)
