from argparse import ArgumentParser, BooleanOptionalAction

import torch

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = ArgumentParser()
    parser.add_argument('--save-logits', '-l', action=BooleanOptionalAction)
    parser.add_argument(
        '--steps', '-s', nargs='+',
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536], type=int
    )
    parser.add_argument('--gpu-id', '-g', type=int, default=6)
    parser.add_argument('--models-path', '-p', type=str)
    parser.add_argument('--models-hidden-sizes', '-z', nargs=4, default=[48, 96, 192, 384], type=int)
    parser.add_argument('--models-per-warp', '-m', type=int, default=32)
    parser.add_argument('--warps', '-w', type=int, default=128)
    parser.add_argument('--models-per-gpu', type=int, default=512)
    parser.add_argument('--save_path', type=str, default='./mega_inference_output/')

    args = parser.parse_args()