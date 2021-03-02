
import os
import cv2
import time
import argparse
import torch
import numpy as np

from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config

from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml", help='Configure tracker')
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True, help='Run in CPU')
    args = parser.parse_args()

    cfg = get_config()
    cfg.merge_from_file(args.config_deepsort)
    use_cuda = args.use_cuda and torch.cuda.is_available()
    torch.set_grad_enabled(False)
    model = build_tracker(cfg, use_cuda=False)

    model.reid = True
    model.extractor.net.eval()

    device = 'cuda'
    output_onnx = 'deepsort.onnx'
    # ------------------------ export -----------------------------
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
    input_names = ['input']
    output_names = ['output']

    input_tensor = torch.randn(1, 3, 128, 64, device=device)

    torch.onnx.export(model.extractor.net.cuda(), input_tensor, output_onnx, export_params=True, verbose=False,
                      input_names=input_names, output_names=output_names, opset_version=10,
                      do_constant_folding=True,
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
