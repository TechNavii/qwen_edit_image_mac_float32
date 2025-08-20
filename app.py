import gradio as gr
import torch
from diffusers import QwenImageEditPipeline
from PIL import Image
import numpy as np
import io
import base64
from typing import Optional, Tuple, Dict, Any
import logging
from pathlib import Path
import json
from config import Config
import os
import warnings

# Suppress MPS warnings for unsupported operations
warnings.filterwarnings("ignore", category=UserWarning, module="torch.mps")

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)


class QwenImageEditor:
    def __init__(self, model_name: str = "Qwen/Qwen-Image-Edit", device: str = None):
        self.model_name = model_name
        self.device, self.dtype = self._select_optimal_device(device)
        self.pipeline = None
        self._initialize_model()
    
    def _select_optimal_device(self, requested_device: Optional[str] = None) -> Tuple[str, torch.dtype]:
        """Select the optimal device and dtype for running the model"""
        
        if requested_device:
            # User explicitly requested a device
            if requested_device == "mps" and torch.backends.mps.is_available():
                logger.info("Using Apple Silicon GPU (MPS) as requested")
                # Use float32 for MPS to avoid type mismatch issues
                return "mps", torch.float32
            elif requested_device == "cuda" and torch.cuda.is_available():
                logger.info("Using NVIDIA GPU (CUDA) as requested")
                return "cuda", torch.bfloat16
            elif requested_device == "cpu":
                logger.info("Using CPU as requested")
                return "cpu", torch.float32
            else:
                logger.warning(f"Requested device {requested_device} not available, auto-selecting...")
        
        # Auto-select best available device
        if torch.backends.mps.is_available():
            # Apple Silicon GPU detected
            logger.info("🎉 Apple Silicon GPU detected! Using MPS for acceleration")
            # Check if MPS is built and functional
            try:
                # Test MPS availability
                test_tensor = torch.tensor([1.0]).to("mps")
                del test_tensor
                # Use float32 for MPS to avoid half precision issues
                return "mps", torch.float32
            except Exception as e:
                logger.warning(f"MPS available but not functional: {e}")
        
        if torch.cuda.is_available():
            # NVIDIA GPU detected
            logger.info("NVIDIA GPU detected! Using CUDA for acceleration")
            return "cuda", torch.bfloat16
        
        # Fallback to CPU
        logger.info("No GPU detected, using CPU (this will be slow)")
        return "cpu", torch.float32
    
    def _initialize_model(self):
        try:
            logger.info(f"Loading Qwen-Image-Edit model on {self.device} with dtype {self.dtype}")
            
            # Set environment variables for optimal MPS performance
            if self.device == "mps":
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable CPU fallback for unsupported ops
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Better memory management
            
            # Load the Qwen-Image-Edit pipeline with device-specific optimizations
            load_kwargs = {
                "torch_dtype": self.dtype,
                "use_safetensors": True,  # Faster loading
            }
            
            self.pipeline = QwenImageEditPipeline.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # Move to device with optimizations
            if self.device == "mps":
                # For MPS, we need to be careful with memory
                self.pipeline = self.pipeline.to(self.device)
                # Enable memory efficient attention if available
                try:
                    self.pipeline.enable_attention_slicing()
                    logger.info("Enabled attention slicing for memory efficiency")
                except:
                    pass
                
                # Enable VAE slicing for large images
                try:
                    self.pipeline.enable_vae_slicing()
                    logger.info("Enabled VAE slicing for large image support")
                except:
                    pass
                    
            elif self.device == "cuda":
                self.pipeline = self.pipeline.to(self.device)
                # Enable xFormers for memory efficiency on CUDA
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xFormers memory efficient attention")
                except:
                    pass
            else:
                # CPU
                self.pipeline = self.pipeline.to(self.device)
                # Enable CPU offload for memory efficiency
                try:
                    self.pipeline.enable_model_cpu_offload()
                    logger.info("Enabled model CPU offload")
                except:
                    pass
            
            # Set up optimized scheduler
            self.pipeline.set_progress_bar_config(disable=False)
            
            logger.info(f"✅ Qwen-Image-Edit model loaded successfully on {self.device}")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Dtype: {self.dtype}")
            logger.info(f"   Memory optimizations: Enabled")
            if self.device == "mps":
                logger.info(f"   Note: Using float32 for MPS to ensure stability")
            
        except Exception as e:
            logger.error(f"Error loading Qwen-Image-Edit model: {e}")
            raise
    
    def process_image(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = " ",
        true_cfg_scale: float = 4.0,
        num_inference_steps: int = 50,
        seed: int = 0
    ) -> Tuple[Optional[Image.Image], str]:
        try:
            if image is None:
                return None, "Please upload an image first"
            
            if not prompt:
                return None, "Please provide editing instructions"
            
            # Ensure image is in RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize image if too large (for memory efficiency)
            max_size = 1024 if self.device == "mps" else 1536
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Resized image to {new_size} for memory efficiency")
            
            # Prepare inputs for Qwen-Image-Edit
            # Create generator with proper device handling
            if self.device == "cpu":
                generator = torch.Generator(device="cpu").manual_seed(seed)
            else:
                # For GPU devices (MPS/CUDA)
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            inputs = {
                "image": image,
                "prompt": prompt,
                "generator": generator,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
            }
            
            # Process image with device-specific optimizations
            # Note: Removed autocast for MPS to avoid type mismatch issues
            try:
                with torch.inference_mode():
                    output = self.pipeline(**inputs)
                    edited_image = output.images[0]
            except RuntimeError as e:
                if "mps" in str(e).lower() and "type" in str(e).lower():
                    logger.error(f"MPS type mismatch error. Trying CPU fallback...")
                    # Try to fallback to CPU if MPS has issues
                    self.pipeline = self.pipeline.to("cpu")
                    self.device = "cpu"
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    inputs["generator"] = generator
                    
                    with torch.inference_mode():
                        output = self.pipeline(**inputs)
                        edited_image = output.images[0]
                    
                    # Move back to MPS after processing
                    self.pipeline = self.pipeline.to("mps")
                    self.device = "mps"
                else:
                    raise
            
            # Clear cache after inference
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            
            return edited_image, f"✅ Successfully edited image with: {prompt}"
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            # Try to clear memory on error
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            return None, f"Error: {str(e)}"
    
    def get_device_info(self) -> str:
        """Get information about the current device"""
        if self.device == "mps":
            return f"🍎 Apple Silicon GPU (Metal Performance Shaders - float32)"
        elif self.device == "cuda":
            return f"🎮 NVIDIA GPU ({torch.cuda.get_device_name()})"
        else:
            return f"💻 CPU (Consider using GPU for faster processing)"


def create_gradio_interface() -> gr.Blocks:
    editor = QwenImageEditor(model_name="Qwen/Qwen-Image-Edit")
    
    custom_css = """
    .gradio-container {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .device-info {
        background: rgba(255,255,255,0.1);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 600;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .header h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .header p {
        font-size: 1.2rem;
        opacity: 0.95;
    }
    .input-group {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .output-group {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    .gr-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        cursor: pointer;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .gr-button-secondary {
        background: #f3f4f6;
        color: #1f2937;
    }
    .gr-button-secondary:hover {
        background: #e5e7eb;
    }
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .slider-container {
        padding: 1rem;
        background: #f9fafb;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .examples-container {
        margin-top: 2rem;
        padding: 1.5rem;
        background: #f9fafb;
        border-radius: 15px;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: white;
        opacity: 0.9;
    }
    .performance-tip {
        background: #fef3c7;
        border: 1px solid #fbbf24;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Qwen-Image-Edit") as interface:
        gr.HTML(f"""
            <div class="header">
                <h1>🎨 Qwen-Image-Edit</h1>
                <p>Advanced AI-powered image editing with natural language instructions</p>
                <div class="device-info">Running on: {editor.get_device_info()}</div>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1, elem_classes="input-group"):
                gr.HTML("<h3 style='margin-bottom: 1rem;'>📥 Input</h3>")
                
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    elem_classes="image-container"
                )
                
                prompt = gr.Textbox(
                    label="Edit Instructions",
                    placeholder="Describe how you want to edit the image (e.g., 'Change the sky to sunset', 'Make the car red', 'Add snow to the scene')",
                    lines=3
                )
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        value=" ",
                        placeholder="What to avoid in the edit",
                        lines=2
                    )
                    
                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=4.0,
                        step=0.5,
                        label="CFG Scale",
                        info="How closely to follow the edit instruction (higher = more adherence)"
                    )
                    
                    steps = gr.Slider(
                        minimum=20,
                        maximum=100,
                        value=30 if editor.device == "mps" else 50,
                        step=5,
                        label="Inference Steps",
                        info="More steps = higher quality but slower (reduced for MPS)"
                    )
                    
                    seed = gr.Number(
                        label="Seed",
                        value=0,
                        precision=0,
                        info="Use same seed for reproducible results"
                    )
                
                # Performance tips for MPS
                if editor.device == "mps":
                    gr.HTML("""
                        <div class="performance-tip">
                            <strong>💡 Apple Silicon GPU Tips:</strong>
                            <ul style='margin: 0.5rem 0;'>
                                <li>Images are auto-resized to max 1024px for optimal performance</li>
                                <li>Use 20-30 steps for faster processing</li>
                                <li>First run may be slower due to compilation</li>
                            </ul>
                        </div>
                    """)
                
                with gr.Row():
                    edit_btn = gr.Button(
                        "🎨 Edit Image",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        "🔄 Clear",
                        variant="secondary",
                        size="lg"
                    )
            
            with gr.Column(scale=1, elem_classes="output-group"):
                gr.HTML("<h3 style='margin-bottom: 1rem;'>📤 Output</h3>")
                
                output_image = gr.Image(
                    label="Edited Image",
                    type="pil",
                    elem_classes="image-container"
                )
                
                output_text = gr.Textbox(
                    label="Processing Status",
                    lines=2,
                    interactive=False
                )
        
        # Examples section
        gr.HTML("""
            <div class="examples-container">
                <h3>💡 Example Edit Instructions</h3>
                <p style='margin-bottom: 1rem;'>Qwen-Image-Edit supports both semantic and appearance editing:</p>
                <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;'>
                    <div>
                        <strong>Semantic Editing:</strong>
                        <ul style='margin-top: 0.5rem;'>
                            <li>Change the rabbit's color to purple</li>
                            <li>Replace the car with a truck</li>
                            <li>Add a rainbow in the sky</li>
                            <li>Remove the person from the image</li>
                            <li>Turn day into night</li>
                            <li>Make it rain in the scene</li>
                        </ul>
                    </div>
                    <div>
                        <strong>Appearance & Text Editing:</strong>
                        <ul style='margin-top: 0.5rem;'>
                            <li>Change the text on the sign to "Welcome"</li>
                            <li>Make the water more blue</li>
                            <li>Add snow on the ground</li>
                            <li>Make the lighting warmer</li>
                            <li>Blur the background</li>
                            <li>Enhance the colors</li>
                        </ul>
                    </div>
                </div>
                <p style='margin-top: 1rem; color: #666;'>
                    <strong>Note:</strong> Qwen-Image-Edit excels at precise text editing while preserving font, size, and style.
                    It supports both English and Chinese text editing.
                </p>
            </div>
        """)
        
        # Example prompts
        gr.Examples(
            examples=[
                ["Change the sky to sunset with orange and pink colors"],
                ["Make the car red and add racing stripes"],
                ["Replace the grass with snow"],
                ["Add 'SALE' text on the banner in red letters"],
                ["Change the building color to blue"],
                ["Remove all people from the scene"],
                ["Make it look like vintage photograph"],
                ["Add fireworks in the sky"],
            ],
            inputs=prompt,
            label="Quick Examples"
        )
        
        def process_edit(image, prompt_text, neg_prompt, cfg, num_steps, seed_val):
            if image is None:
                return None, "Please upload an image first"
            if not prompt_text:
                return None, "Please provide editing instructions"
            
            return editor.process_image(
                image, 
                prompt_text, 
                neg_prompt, 
                cfg, 
                num_steps, 
                int(seed_val)
            )
        
        def clear_all():
            default_steps = 30 if editor.device == "mps" else 50
            return None, "", None, "", " ", 4.0, default_steps, 0
        
        edit_btn.click(
            fn=process_edit,
            inputs=[input_image, prompt, negative_prompt, cfg_scale, steps, seed],
            outputs=[output_image, output_text]
        )
        
        clear_btn.click(
            fn=clear_all,
            outputs=[input_image, prompt, output_image, output_text, negative_prompt, cfg_scale, steps, seed]
        )
        
        gr.HTML(f"""
            <div class="footer">
                <p>Powered by Qwen-Image-Edit Model | Built with ❤️ using Gradio</p>
                <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
                    Device: {editor.get_device_info()} | 
                    Performance: {"⚡ Fast" if editor.device in ["mps", "cuda"] else "🐌 Slow (CPU)"}
                </p>
            </div>
        """)
    
    return interface


def cleanup_memory():
    """Force memory cleanup"""
    import gc
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            torch.mps.synchronize()
        except:
            pass
    
    # Force garbage collection
    gc.collect()
    
    logger.info("✅ Memory cleanup completed")


def main():
    # Print device information at startup
    test_editor = QwenImageEditor(model_name="Qwen/Qwen-Image-Edit")
    print("\n" + "="*60)
    print(f"🚀 Starting Qwen-Image-Edit Server")
    print(f"📱 Device: {test_editor.get_device_info()}")
    print(f"🎯 Performance: {'⚡ GPU Accelerated' if test_editor.device in ['mps', 'cuda'] else '🐌 CPU Mode (Slow)'}")
    print("="*60 + "\n")
    
    import atexit
    import signal
    import sys
    
    # Register cleanup on exit
    def cleanup_on_exit():
        print("\n🧹 Cleaning up memory...")
        cleanup_memory()
        # Delete the model
        if 'test_editor' in locals():
            del test_editor
        cleanup_memory()
    
    atexit.register(cleanup_on_exit)
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n🛑 Interrupted, cleaning up...")
        cleanup_on_exit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except KeyboardInterrupt:
        cleanup_on_exit()
    except Exception as e:
        print(f"Error: {e}")
        cleanup_on_exit()


if __name__ == "__main__":
    main()