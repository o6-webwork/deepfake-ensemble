import streamlit as st
import torch
import numpy as np
import logging
import tempfile
import shutil
import os
import cv2  # Required for blending
from PIL import Image
from pathlib import Path

# Import SPAI internals
from spai.config import get_config
from spai.models import build_cls_model
from spai.utils import load_pretrained
from spai.data.data_finetune import build_transform

# Configure Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spai_ui")

# Constants
CONFIG_PATH = "configs/spai.yaml"
WEIGHTS_PATH = "weights/spai.pth"
OUTPUT_DIR = "output"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.set_page_config(page_title="SPAI Detector", layout="wide")

@st.cache_resource
def load_model(device_name):
    if not Path(WEIGHTS_PATH).exists():
        st.error(f"Weights file not found at {WEIGHTS_PATH}. Please download it.")
        return None, None

    config = get_config({
        "cfg": CONFIG_PATH,
        "batch_size": 1,
        "opts": []
    })

    model = build_cls_model(config)
    model.to(device_name)
    
    load_pretrained(config, model, logger, checkpoint_path=WEIGHTS_PATH, verbose=True)
    model.eval()
    
    return model, config

def create_transparent_overlay(original_pil, overlay_path, alpha=0.6):
    """
    Manually blends the original image with the heatmap/overlay to ensure transparency.
    """
    # 1. Convert Original PIL to OpenCV format (RGB)
    background = np.array(original_pil)
    
    # 2. Load the Overlay/Heatmap from disk
    foreground = cv2.imread(str(overlay_path))
    
    if foreground is None:
        return background # Fallback if read fails

    # Convert BGR (OpenCV default) to RGB to match PIL
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)

    # 3. Resize Foreground to match Background exactly
    # (cv2.resize expects (width, height))
    if foreground.shape[:2] != background.shape[:2]:
        foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

    # 4. Blend
    # alpha: Weight of the original image
    # beta (1-alpha): Weight of the heatmap
    beta = 1.0 - alpha
    blended = cv2.addWeighted(background, alpha, foreground, beta, 0)
    
    return blended

def main():
    st.title("🕵️ SPAI: AI-Generated Image Detector")
    st.markdown("Upload an image to check if it is Real or AI-Generated.")

    # Sidebar Options
    st.sidebar.header("Settings")
    device_opt = "cuda" if torch.cuda.is_available() else "cpu"
    st.sidebar.write(f"**Device:** {device_opt.upper()}")
    
    max_size = st.sidebar.select_slider(
        "Max Resolution (Longest Edge)",
        options=[512, 768, 1024, 1280, 1536, 2048, "Original"],
        value=1280,
    )
    
    show_attention = st.sidebar.checkbox("Generate Attention Heatmap", value=True)
    
    # New Slider: Transparency Control
    overlay_alpha = st.sidebar.slider("Overlay Transparency (Original vs Heatmap)", 0.0, 1.0, 0.6)

    with st.spinner("Loading Model..."):
        model, base_config = load_model(device_opt)

    if model is None:
        return

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            # We keep the original PIL image for blending later
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"Uploaded Image ({image.width}x{image.height})", use_column_width=True)

        with col2:
            st.write("Analyzing...")
            progress_bar = st.progress(0)
            
            try:
                # Setup Config
                config = base_config.clone()
                config.defrost()
                config.TEST.MAX_SIZE = int(max_size) if max_size != "Original" else None
                if show_attention:
                    config.MODEL.RESOLUTION_MODE = "arbitrary" 
                config.freeze()

                if device_opt == "cuda":
                    torch.cuda.empty_cache()
                
                progress_bar.progress(20)

                # Preprocess
                transform = build_transform(is_train=False, config=config)
                img_np = np.array(image)
                input_tensor = transform(image=img_np)["image"]
                input_batch = input_tensor.unsqueeze(0).to(device_opt)
                
                progress_bar.progress(50)

                # Create temp dir for heatmaps
                with tempfile.TemporaryDirectory() as temp_export_dir:
                    export_path = Path(temp_export_dir)
                    
                    with torch.no_grad():
                        if show_attention and config.MODEL.RESOLUTION_MODE == "arbitrary":
                            output, attention_masks = model(
                                x=[input_batch], 
                                feature_extraction_batch_size=config.MODEL.FEATURE_EXTRACTION_BATCH,
                                export_dirs=[export_path]
                            )
                        else:
                            attention_masks = None
                            if config.MODEL.RESOLUTION_MODE == "arbitrary":
                                output = model([input_batch], config.MODEL.FEATURE_EXTRACTION_BATCH)
                            else:
                                output = model(input_batch)
                    
                    progress_bar.progress(90)

                    # Result
                    score = torch.sigmoid(output).item()
                    
                    st.divider()
                    st.metric(label="AI-Generated Probability", value=f"{score:.2%}")
                    
                    if score > 0.5:
                        st.error("🚨 Classification: **AI-Generated**")
                    else:
                        st.success("✅ Classification: **Real Image**")
                    
                    # --- HEATMAP PROCESSING ---
                    if attention_masks:
                        mask_obj = attention_masks[0]
                        
                        # Check if the file exists in the temp directory
                        if hasattr(mask_obj, 'overlay') and mask_obj.overlay and mask_obj.overlay.exists():
                            st.subheader("Spectral Attention Map")
                            st.write("Red/High areas indicate spectral inconsistencies (fake artifacts).")
                            
                            # 1. Create Manually Blended Image
                            # We pass the original PIL 'image' and the path to the model's output
                            blended_img = create_transparent_overlay(
                                image, 
                                mask_obj.overlay, 
                                alpha=overlay_alpha
                            )
                            
                            # 2. Display in Streamlit
                            st.image(blended_img, caption="Custom Transparent Overlay", use_column_width=True)

                            # 3. Save Logic
                            # Convert numpy blended image back to bytes for download/saving
                            # We use cv2.imencode to get bytes
                            blended_bgr = cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR)
                            success, buffer = cv2.imencode(".jpg", blended_bgr)
                            
                            if success:
                                byte_data = buffer.tobytes()
                                
                                # A. Save Local
                                save_filename = f"blended_{uploaded_file.name}"
                                local_path = Path(OUTPUT_DIR) / save_filename
                                with open(local_path, "wb") as f:
                                    f.write(byte_data)
                                st.caption(f"Blended image saved to: `{local_path}`")

                                # B. Download Button
                                st.download_button(
                                    label="📥 Download Blended Image",
                                    data=byte_data,
                                    file_name=save_filename,
                                    mime="image/jpeg"
                                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.exception("Inference failed")
            finally:
                progress_bar.empty()

if __name__ == "__main__":
    main()