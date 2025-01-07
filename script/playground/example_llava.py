import argparse

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

import copy
import warnings
from decord import VideoReader, cpu
import numpy as np
import torch
import pandas as pd
import torchvision.transforms as T
import os

from framefusion.interface import apply_framefusion
from framefusion.utils import save_video_frames

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Video processing and analysis script")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="lmms-lab/LLaVA-Video-7B-Qwen2",
        help="Pretrained model path",
    )
    parser.add_argument("--model-name", type=str, default="llava_qwen", help="Model name")
    parser.add_argument(
        "--conv-template",
        type=str,
        default="qwen_1_5",
        help="Chat template for the model",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--device-map", type=str, default="auto", help="Device mapping strategy")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", help="Torch data type")

    # FrameFusion arguments
    parser.add_argument("--framefusion-cost", type=float, default=0.3, help="FrameFusion cost")
    parser.add_argument("--framefusion-similarity-lower-bound", type=float, default=0.6, help="FrameFusion similarity lower bound")
    parser.add_argument("--framefusion-ratio-lower-bound", type=float, default=0.1, help="FrameFusion ratio lower bound")

    # Data arguments
    parser.add_argument(
        "--video-path",
        type=str,
        default="example/video/Tom_Jerry.mp4",
        help="Path to the input video",
    )
    parser.add_argument("--max-frames", type=int, default=64, help="Maximum number of frames to process")

    # Output arguments
    parser.add_argument(
        "--save-video-frames",
        action="store_true",
        help="Save video frames to local folder",
    )

    return parser.parse_args()


def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time, video_time

if __name__ == "__main__":
    # get args
    args = parse_args()

    # load model
    dtype = getattr(torch, args.torch_dtype)
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        torch_dtype=args.torch_dtype,
        attn_implementation="sdpa",
        device_map=args.device_map,
    )
    model.eval()

    # load video
    video_path = args.video_path
    max_frames_num = args.max_frames
    video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
    video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].to(args.device).to(dtype)
    video = [video]

    # save the video frames to local folder
    if args.save_video_frames:
        save_video_frames(video, output_path="local/video_frames")

    # prompts
    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}. Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"{time_instruciton}\n Which animal hit the cat? Answer it simply."

    # compression configs
    config_dict = {
        "dense": dict(),
        "framefusion": {
            "cost": args.framefusion_cost,
            "similarity_lower_bound": args.framefusion_similarity_lower_bound,
            "ratio_lower_bound": args.framefusion_ratio_lower_bound,
        },
    }

    # run experiments
    results = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for name, config in config_dict.items():
        # compress the model
        if name == "framefusion":
            apply_framefusion(model, **config)

        # prepare the conversation
        conv = copy.deepcopy(conv_templates[args.conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(args.device)

        # prefetch for accurate time measurement
        response = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=1024,
        )

        # generate the response
        start.record()
        response = model.generate(
            input_ids,
            images=video,
            modalities=["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=1024,
        )
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)

        # decode the response
        text_outputs = tokenizer.batch_decode(response, skip_special_tokens=True)[0].strip()
        print("\n\n", text_outputs, "\n\n")

        # save the results
        results.append(
            {
                "method": name,
                "cost": config.get("cost", 1.0),
                "similarity_lower_bound": str(config.get("similarity_lower_bound", "-")),
                "ratio_lower_bound": str(config.get("ratio_lower_bound", "-")),
                "text_outputs": text_outputs,
                "time": time,
            }
        )

    # print results as a table with pandas
    df = pd.DataFrame(results)
    print(df)
