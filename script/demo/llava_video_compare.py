"""Simplified Gradio demo for FrameFusion vs LLaVA-Video comparison."""

import os
import time
import copy
import threading
from typing import Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure HuggingFace offline mode (but allow Gradio to access internet for share functionality)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Comment out the proxy deletion to allow Gradio share to work
if "http_proxy" in os.environ:
    del os.environ["http_proxy"]
if "https_proxy" in os.environ:
    del os.environ["https_proxy"]
if "HTTP_PROXY" in os.environ:
    del os.environ["HTTP_PROXY"]
if "HTTPS_PROXY" in os.environ:
    del os.environ["HTTPS_PROXY"]

import gradio as gr
import numpy as np
import pandas as pd
import torch
import socket

try:
    from decord import VideoReader, cpu
except Exception:
    VideoReader = None
    cpu = None

# LLaVA-Video (NeXT) imports
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
except Exception:
    load_pretrained_model = None
    tokenizer_image_token = None
    IMAGE_TOKEN_INDEX = None
    DEFAULT_IMAGE_TOKEN = "<image>"
    conv_templates = None

from framefusion.interface import apply_framefusion

# ----------------------
# Utilities
# ----------------------
def find_available_port(start_port=7860, max_tries=10):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_tries):
        try:
            # Check if port is available
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('0.0.0.0', port))
                s.listen(1)
                print(f"Port {port} is available")
                return port
        except OSError as e:
            print(f"Port {port} is busy: {e}")
            continue
    print(f"No available ports found in range {start_port}-{start_port + max_tries - 1}, using {start_port}")
    return start_port  # fallback to original port


def load_video_np(path: str, max_frames: int = 128) -> Optional[np.ndarray]:
    """Load and uniformly sample frames. Returns None if failed."""
    if VideoReader is None:
        print("Error: decord not installed")
        return None
    
    try:
        vr = VideoReader(path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        
        # Uniform sample to max_frames
        if total_frame_num > max_frames:
            frame_idx = np.linspace(0, total_frame_num - 1, max_frames, dtype=int).tolist()
        else:
            frame_idx = list(range(total_frame_num))
        
        frames = vr.get_batch(frame_idx).asnumpy()  # (N, H, W, C) uint8
        return frames
    except Exception as e:
        print(f"Failed to load video: {e}")
        return None


class SimpleRunner:
    """Simplified model runner."""
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.is_loaded = False
        
    def load(self):
        """Load the model."""
        if self.is_loaded:
            return
            
        if load_pretrained_model is None:
            print("Error: LLaVA not installed")
            return
            
        try:
            print(f"Loading model {self.model_path} on {self.device}...")
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                self.model_path,
                None,
                "llava_qwen",
                torch_dtype="bfloat16",
                attn_implementation="sdpa",
                device_map=self.device,
            )
            self.model.eval()
            
            # Ensure the entire model is on the specified device
            if torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.is_loaded = True
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.is_loaded = False
    
    @torch.inference_mode()
    def generate(self, video_np: Optional[np.ndarray], prompt: str, max_new_tokens: int = 128) -> Tuple[str, float]:
        """Generate response. Returns (text, time_ms)."""
        if not self.is_loaded:
            return "Model not loaded", 0.0
            
        try:
            # Use the device specified during initialization
            device = torch.device(self.device)
            dtype = next(self.model.parameters()).dtype
            
            # Prepare inputs
            if video_np is not None:
                pixel_values = self.image_processor.preprocess(video_np, return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(device).to(dtype)
                images = [pixel_values]
                modalities = ["video"]
                question = f"{DEFAULT_IMAGE_TOKEN} {prompt}"
            else:
                # Create a dummy blank image for text-only generation
                # LLaVA models expect images, so we provide a minimal one
                dummy_image = np.zeros((1, 224, 224, 3), dtype=np.uint8)  # Single black frame
                pixel_values = self.image_processor.preprocess(dummy_image, return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(device).to(dtype)
                images = [pixel_values]
                modalities = ["video"]
                # Still include image token even for dummy image
                question = f"{DEFAULT_IMAGE_TOKEN} {prompt}"
            
            # Build conversation
            conv = copy.deepcopy(conv_templates["qwen_1_5"])
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            
            # Tokenization - always use tokenizer_image_token since we always have an image token
            input_ids = tokenizer_image_token(
                prompt_question,
                self.tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt"
            ).unsqueeze(0).to(device)
            
            # Generate - always include images and modalities (following example_llava.py)
            t0 = time.time()
            
            outputs = self.model.generate(
                input_ids,
                images=images,
                modalities=modalities,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
            )
            elapsed_ms = (time.time() - t0) * 1000
            
            text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            return text, elapsed_ms
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {e}", 0.0


# Global runners
RUNNER_BASE = None
RUNNER_FF = None


def init_models():
    """Initialize both model instances."""
    global RUNNER_BASE, RUNNER_FF
    
    model_path = "lmms-lab/LLaVA-Video-7B-Qwen2"
    
    print("Initializing models...")
    
    # Baseline model
    RUNNER_BASE = SimpleRunner(model_path, "cuda:0")
    RUNNER_BASE.load()
    
    # FrameFusion model
    RUNNER_FF = SimpleRunner(model_path, "cuda:1")
    RUNNER_FF.load()
    
    if RUNNER_FF.is_loaded:
        # Apply FrameFusion
        apply_framefusion(
            RUNNER_FF.model,
            cost=0.3,
            similarity_lower_bound=0.6,
            ratio_lower_bound=0.1,
        )
        print("FrameFusion applied to second model")
    
    return RUNNER_BASE, RUNNER_FF


def simple_compare(video_file, prompt: str, max_frames: int, max_new_tokens: int):
    """Simplified comparison function that yields results as they complete."""
    print("\n" + "="*60)
    print("simple_compare called")
    print(f"  video_file type: {type(video_file)}")
    print(f"  prompt: {prompt}")
    print(f"  max_frames: {max_frames}")
    print(f"  max_new_tokens: {max_new_tokens}")
    
    # Initial state - show "Running..." for both
    yield ("🔄 Running...", 0.0, "🔄 Running...", 0.0)
    
    if RUNNER_BASE is None or RUNNER_FF is None:
        print("Models not initialized")
        yield ("❌ Models not initialized", 0.0, "❌ Models not initialized", 0.0)
        return
    
    if not RUNNER_BASE.is_loaded or not RUNNER_FF.is_loaded:
        print("Models not loaded")
        yield ("❌ Models not loaded", 0.0, "❌ Models not loaded", 0.0)
        return
    
    # Handle video input
    frames = None
    if video_file is not None:
        video_path = None
        
        # Extract path from various formats
        if isinstance(video_file, str) and video_file:
            video_path = video_file
        elif hasattr(video_file, 'name'):
            video_path = video_file.name
        elif isinstance(video_file, dict):
            video_path = video_file.get('path') or video_file.get('name')
        
        print(f"  Resolved video_path: {video_path}")
        
        # Load video if we have a path
        if video_path and os.path.exists(video_path):
            frames = load_video_np(video_path, max_frames)
            if frames is not None:
                print(f"  Loaded {len(frames)} frames")
            else:
                print(f"  Failed to load video")
        else:
            print(f"  No valid video path")
    else:
        print(f"  No video provided")
    
    # Ensure we have a prompt
    if not prompt or not prompt.strip():
        prompt = "Describe this." if frames is not None else "Hello!"
    
    print(f"  Using prompt: {prompt}")
    
    try:
        # Run both models in parallel
        print("  Running both models in parallel...")
        
        def run_baseline():
            print("    Baseline starting...")
            result = RUNNER_BASE.generate(frames, prompt, max_new_tokens)
            print(f"    Baseline done: {result[1]:.1f}ms")
            return ('baseline', result)
        
        def run_framefusion():
            print("    FrameFusion starting...")
            result = RUNNER_FF.generate(frames, prompt, max_new_tokens)
            print(f"    FrameFusion done: {result[1]:.1f}ms")
            return ('framefusion', result)
        
        # Execute both functions in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future_base = executor.submit(run_baseline)
            future_ff = executor.submit(run_framefusion)
            
            # Track results as they complete
            base_text, base_time = "🔄 Running...", 0.0
            ff_text, ff_time = "🔄 Running...", 0.0
            
            # Wait for results and yield as they complete
            for future in as_completed([future_base, future_ff]):
                model_type, (text, time_ms) = future.result()
                
                if model_type == 'baseline':
                    base_text, base_time = str(text), float(time_ms)
                else:  # framefusion
                    ff_text, ff_time = str(text), float(time_ms)
                
                # Yield current state after each completion
                yield (base_text, base_time, ff_text, ff_time)
        
        print("  Both models completed")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"Error in comparison: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        yield (f"❌ Error: {e}", 0.0, f"❌ Error: {e}", 0.0)


def build_simple_demo():
    """Build clean and compact Gradio interface."""
    
    with gr.Blocks(title="FrameFusion Demo", theme=gr.themes.Base()) as demo:
        # Header
        gr.Markdown("# 🚀 FrameFusion Demo")
        gr.Markdown("Compare video understanding performance and efficiency")
        
        # ===== SECTION 1: SETUP =====
        gr.Markdown("## 📝 Section 1: Setup")
        with gr.Row():
            # Left: Video Input
            with gr.Column(scale=1):
                video_input = gr.Video(label="Video Input", height=300)
                
            # Right: Prompt and Settings
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Prompt",
                    value="Hello.",
                    lines=3
                )
                with gr.Row():
                    max_frames_slider = gr.Slider(
                        minimum=8,
                        maximum=128,
                        value=32,
                        step=8,
                        label="Max Frames"
                    )
                    max_tokens_slider = gr.Slider(
                        minimum=16,
                        maximum=256,
                        value=128,
                        step=16,
                        label="Max Tokens"
                    )
        
        # Examples in Section 1
        example_video = "example/video/Tom_Jerry.mp4"
        if os.path.exists(example_video):
            gr.Examples(
                examples=[[example_video, "Which animal hit the cat?", 128, 128]],
                inputs=[video_input, prompt_input, max_frames_slider, max_tokens_slider],
                label="Example:"
            )
        
        # ===== SECTION 2: RUN =====
        gr.Markdown("## 🚀 Section 2: Run")
        
        # Run Button
        run_btn = gr.Button("Start Comparison", variant="primary", size="lg")
        
        # Results directly under the button
        gr.Markdown("### Results")
        with gr.Row():
            # Original Model Column
            with gr.Column():
                gr.Markdown("**Original Model**")
                base_progress = gr.HTML(value="")
                base_output = gr.Textbox(
                    label="Output",
                    lines=4,
                    max_lines=6,
                    placeholder="Original model output will appear here..."
                )
                base_time_output = gr.Number(label="Time (ms)", precision=1)
            
            # FrameFusion Model Column  
            with gr.Column():
                gr.Markdown("**FrameFusion Optimized**")
                ff_progress = gr.HTML(value="")
                ff_output = gr.Textbox(
                    label="Output",
                    lines=4,
                    max_lines=6,
                    placeholder="FrameFusion output will appear here..."
                )
                ff_time_output = gr.Number(label="Time (ms)", precision=1)
        
        # Performance Visualization Section
        gr.Markdown("### Performance Comparison")
        performance_chart = gr.BarPlot(
            title="Runtime Comparison",
            x="model",
            y="time",
            y_title="Processing Time (ms)",
            height=300,
            y_lim=[0, None],
            color_map={"Original LLaVA-Video": "#3b82f6", "FrameFusion Optimized": "#10b981"}
        )
        performance_display = gr.HTML(value="⏳ Ready to compare")
        
        # Enhanced click handler with performance metrics and visualization
        def enhanced_compare(*args):
            for result in simple_compare(*args):
                base_text, base_time, ff_text, ff_time = result
                
                # Create progress bars for each model
                if base_text.startswith("🔄"):
                    base_progress_html = '''
                    <div style="width: 100%; background-color: #f0f0f0; border-radius: 12px; padding: 3px; margin: 5px 0;">
                        <div style="width: 50%; background-color: #007bff; height: 30px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px; font-weight: bold;">
                            Running...
                        </div>
                    </div>
                    '''
                    base_clean_text = ""
                elif base_time > 0:
                    base_progress_html = '''
                    <div style="width: 100%; background-color: #f0f0f0; border-radius: 12px; padding: 3px; margin: 5px 0;">
                        <div style="width: 100%; background-color: #28a745; height: 30px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px; font-weight: bold;">
                            ✅ Completed
                        </div>
                    </div>
                    '''
                    base_clean_text = base_text
                else:
                    base_progress_html = ""
                    base_clean_text = base_text
                
                if ff_text.startswith("🔄"):
                    ff_progress_html = '''
                    <div style="width: 100%; background-color: #f0f0f0; border-radius: 12px; padding: 3px; margin: 5px 0;">
                        <div style="width: 50%; background-color: #007bff; height: 30px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px; font-weight: bold;">
                            Running...
                        </div>
                    </div>
                    '''
                    ff_clean_text = ""
                elif ff_time > 0:
                    ff_progress_html = '''
                    <div style="width: 100%; background-color: #f0f0f0; border-radius: 12px; padding: 3px; margin: 5px 0;">
                        <div style="width: 100%; background-color: #28a745; height: 30px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 14px; font-weight: bold;">
                            ✅ Completed
                        </div>
                    </div>
                    '''
                    ff_clean_text = ff_text
                else:
                    ff_progress_html = ""
                    ff_clean_text = ff_text
                
                # Only show chart when both models have finished
                if base_time > 0 and ff_time > 0:
                    # Both models finished - create DataFrame for BarPlot
                    chart_data = pd.DataFrame({
                        'model': ['Original Model', 'FrameFusion Optimized'],
                        'time': [base_time, ff_time]
                    })
                    
                    # Calculate metrics
                    speedup = base_time / ff_time
                    time_saved = base_time - ff_time
                    
                    metrics_html = f'''
                    <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                        <strong>Performance:</strong> {speedup:.2f}x speedup | {time_saved:.0f}ms saved
                    </div>
                    '''
                else:
                    # Models still processing - don't show chart yet
                    chart_data = None
                    metrics_html = '<div style="text-align: center; padding: 10px;">⏳ Waiting for both models to complete...</div>'
                
                yield base_progress_html, base_clean_text, base_time, ff_progress_html, ff_clean_text, ff_time, chart_data, metrics_html
        
        # Connect the enhanced button handler
        run_btn.click(
            fn=enhanced_compare,
            inputs=[video_input, prompt_input, max_frames_slider, max_tokens_slider],
            outputs=[base_progress, base_output, base_time_output, ff_progress, ff_output, ff_time_output, performance_chart, performance_display],
            stream_every=0.5
        )
        

    
    return demo


def test_sanity():
    """Test basic functionality."""
    print("\nRunning sanity test...")
    
    def test_baseline():
        if RUNNER_BASE and RUNNER_BASE.is_loaded:
            text, ms = RUNNER_BASE.generate(None, "Say hello", 16)
            print(f"Baseline test: {text} ({ms:.1f}ms)")
            return text, ms
        return "Not loaded", 0.0
    
    def test_framefusion():
        if RUNNER_FF and RUNNER_FF.is_loaded:
            text, ms = RUNNER_FF.generate(None, "Say hello", 16)
            print(f"FrameFusion test: {text} ({ms:.1f}ms)")
            return text, ms
        return "Not loaded", 0.0
    
    # Run tests in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_base = executor.submit(test_baseline)
        future_ff = executor.submit(test_framefusion)
        
        # Wait for both to complete
        future_base.result()
        future_ff.result()
    
    print("Sanity test complete\n")


if __name__ == "__main__":
    # Initialize models
    init_models()
    
    # Run sanity test
    test_sanity()
    
    # Launch Gradio
    demo = build_simple_demo()
    
    # Find available port and launch the demo
    available_port = find_available_port(7860)
    print(f"Starting Gradio demo on port {available_port}...")
    
    demo.launch(
        share=True, 
        server_name="0.0.0.0", 
        server_port=available_port,
        debug=False,  # Disable debug for cleaner output
        quiet=False,
        show_error=True
    )
