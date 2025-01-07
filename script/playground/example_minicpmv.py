import argparse
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu
import pandas as pd

from framefusion.interface import apply_framefusion


def parse_args():
    parser = argparse.ArgumentParser(description="Video processing and analysis script")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="openbmb/MiniCPM-V-2_6",
        help="Pretrained model path",
    )
    parser.add_argument(
        "--conv-template",
        type=str,
        default="qwen_1_5",
        help="Chat template for the model",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on")
    parser.add_argument("--torch-dtype", type=str, default="bfloat16", help="Torch data type")

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

    # Data arguments
    parser.add_argument(
        "--video-path",
        type=str,
        default="example/video/Tom_Jerry.mp4",
        help="Path to the input video",
    )
    parser.add_argument("--max-frames", type=int, default=64, help="Maximum number of frames to process")

    return parser.parse_args()


def encode_video(video_path, max_frames_num):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_frames_num:
        frame_idx = uniform_sample(frame_idx, max_frames_num)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype("uint8")) for v in frames]
    print("num frames:", len(frames))
    return frames


if __name__ == "__main__":
    # get args
    args = parse_args()

    # load model
    dtype = getattr(torch, args.torch_dtype)
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, attn_implementation="sdpa", torch_dtype=dtype)  # sdpa or flash_attention_2, no eager
    model = model.eval().to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # load video
    frames = encode_video(args.video_path, args.max_frames)

    # prompts
    question = "Which animal hit the cat? Answer it simply."
    # question = "What are the two animals that touch the bone? Answer it simply."
    msgs = [
        {"role": "user", "content": frames + [question]},
    ]

    # Set decode params for video
    params = {"do_sample": False}
    params["use_image_id"] = False
    params["max_slice_nums"] = 2  # use 1 if cuda OOM and video resolution >  448*448

    model.num_frames = args.max_frames

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

        # prefetch for accurate time measurement
        text_outputs = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            max_inp_length=32768,
            max_new_tokens=2,
            **params,
        )

        # generate the response
        start.record()
        text_outputs = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer,
            max_inp_length=32768,
            **params,
        )
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)

        # decode the response
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
