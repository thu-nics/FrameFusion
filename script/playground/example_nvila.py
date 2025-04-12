import argparse
import torch
from termcolor import colored
import os

import llava
from llava import conversation as clib
from llava.media import Image, Video
from framefusion.interface import apply_framefusion

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", "-m", type=str, default="Efficient-Large-Model/NVILA-8B-Video")
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--text", type=str)
    parser.add_argument("--media", type=str, nargs="+", default="example/video/Tom_Jerry.mp4")

    # FrameFusion arguments
    parser.add_argument("--framefusion-cost", type=float, default=0.3, help="FrameFusion cost")
    parser.add_argument(
        "--framefusion-similarity-lower-bound",
        type=float,
        default=0.7,
        help="FrameFusion similarity lower bound",
    )
    parser.add_argument(
        "--framefusion-ratio-lower-bound",
        type=float,
        default=0.1,
        help="FrameFusion ratio lower bound",
    )
    args = parser.parse_args()

    # Load model
    model = llava.load(args.model_path)

    config = {
            "cost": args.framefusion_cost,
            "similarity_lower_bound": args.framefusion_similarity_lower_bound,
            "ratio_lower_bound": args.framefusion_ratio_lower_bound,
        }
    apply_framefusion(model, **config)

    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()


    def get_first_filenames(directory, file_num):
        all_files = os.listdir(directory)
        filenames = sorted([os.path.join(directory, f) for f in all_files if os.path.isfile(os.path.join(directory, f))])
        return filenames[:file_num]

    prompt = []
    if args.media is not None:
        media = args.media

        if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
            media = Image(media)
        elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
            media = Video(media)
        else:
            raise ValueError(f"Unsupported media type: {media}")
        prompt.append(media)
    if args.text is not None:
        prompt.append(args.text)


    response = model.generate_content(prompt)
    print(colored(response, "cyan", attrs=["bold"]))
    
    

if __name__ == "__main__":
    main()