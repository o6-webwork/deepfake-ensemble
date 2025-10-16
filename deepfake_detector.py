import streamlit as st
import requests
import base64
from PIL import Image
import cv2
import tempfile
from typing import List, Optional, Dict
import os
import pandas as pd
import io
from datetime import datetime

st.set_page_config(
    page_title="Deepfake Detection VLM",
    page_icon="🔍",
    layout="wide"
)

def encode_image_to_base64(image_data: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(image_data).decode('utf-8')

def extract_video_frames(video_file, num_frames: int = 5) -> List[bytes]:
    """Extract frames from video file."""
    frames = []

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_file_path = tmp_file.name

    try:
        cap = cv2.VideoCapture(tmp_file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                frames.append(buffer.tobytes())

        cap.release()
    finally:
        os.unlink(tmp_file_path)

    return frames

def get_available_models(endpoint_url: str) -> List[str]:
    """Get available models from the vLLM endpoint."""
    try:
        response = requests.get(
            f"{endpoint_url}/v1/models",
            timeout=10
        )
        response.raise_for_status()
        models_data = response.json()
        return [model["id"] for model in models_data["data"]]
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return []

def call_vllm_api_with_images(images: List[str], user_prompt: str, assistant_prefill: str, temperature: float = 0.0, endpoint_url: str = "http://100.64.0.3:8001", model_name: str = "openbmb/MiniCPM-V-4_5") -> Optional[str]:
    """Call the vLLM OpenAI API endpoint with images."""

    content = [{"type": "text", "text": user_prompt}]

    for img_b64 in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
        })

    messages = [
        {"role": "user", "content": content}
    ]

    if assistant_prefill.strip():
        messages.append({"role": "assistant", "content": assistant_prefill})

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 500,
        "temperature": temperature
    }

    try:
        response = requests.post(
            f"{endpoint_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        response_content = result["choices"][0]["message"]["content"]

        # If we used a prefill, combine it with the response
        if assistant_prefill.strip():
            return assistant_prefill + response_content
        else:
            return response_content

    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return None

def call_vllm_api_text_only(user_prompt: str, temperature: float = 0.0, endpoint_url: str = "http://100.64.0.3:8001", model_name: str = "openbmb/MiniCPM-V-4_5") -> Optional[str]:
    """Call the vLLM OpenAI API endpoint with text only."""

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 100,
        "temperature": temperature
    }

    try:
        response = requests.post(
            f"{endpoint_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]

    except Exception as e:
        st.error(f"Classification API call failed: {str(e)}")
        return None

def analyze_images(images: List[str], user_prompt: str, assistant_prefill: str, temperature: float = 0.0, endpoint_url: str = "http://100.64.0.3:8001", model_name: str = "openbmb/MiniCPM-V-4_5") -> Optional[str]:
    """Step 1: Analyze the images and generate explanation."""
    return call_vllm_api_with_images(images, user_prompt, assistant_prefill, temperature, endpoint_url, model_name)

def classify_analysis(analysis: str, temperature: float = 0.0, endpoint_url: str = "http://100.64.0.3:8001", model_name: str = "openbmb/MiniCPM-V-4_5") -> Optional[str]:
    """Step 2: Classify the analysis as Real or AI Generated."""
    classification_prompt = f"""Based on the following analysis of an image, classify it as either "Real" or "AI Generated".

Analysis: {analysis}

Provide only the classification: Real or AI Generated"""

    return call_vllm_api_text_only(classification_prompt, temperature, endpoint_url, model_name)

def parse_classification(classification_response: str) -> str:
    """Parse the classification response to extract clean classification."""
    if not classification_response:
        return "Unknown"

    response_lower = classification_response.lower().strip()

    if "real" in response_lower and "ai generated" not in response_lower:
        return "Real"
    elif "ai generated" in response_lower or "ai" in response_lower:
        return "AI Generated"
    else:
        return "Unknown"

def create_export_data(results: List[Dict]) -> pd.DataFrame:
    """Create a DataFrame for CSV export."""
    export_data = []
    for result in results:
        export_data.append({
            'filename': result['filename'],
            'endpoint_url': result['endpoint_url'],
            'model_name': result['model_name'],
            'prompt': result['prompt'],
            'assistant_prefill': result['assistant_prefill'],
            'temperature': result['temperature'],
            'analysis': result['analysis'],
            'classification': result['classification'],
            'classification_response': result['classification_response'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    return pd.DataFrame(export_data)

def analyze_single_image(image_b64: str, filename: str, user_prompt: str, assistant_prefill: str, temperature: float, endpoint_url: str, model_name: str) -> Dict:
    """Analyze a single image and return results."""
    # Step 1: Analyze the image
    analysis = analyze_images([image_b64], user_prompt, assistant_prefill, temperature, endpoint_url, model_name)

    if not analysis:
        return {
            'filename': filename,
            'endpoint_url': endpoint_url,
            'model_name': model_name,
            'prompt': user_prompt,
            'assistant_prefill': assistant_prefill,
            'temperature': temperature,
            'analysis': 'Failed to get analysis',
            'classification': 'Error',
            'classification_response': 'Error'
        }

    # Step 2: Classify the analysis
    classification_response = classify_analysis(analysis, temperature, endpoint_url, model_name)

    if not classification_response:
        classification = 'Error'
        classification_response = 'Failed to get classification'
    else:
        classification = parse_classification(classification_response)

    return {
        'filename': filename,
        'endpoint_url': endpoint_url,
        'model_name': model_name,
        'prompt': user_prompt,
        'assistant_prefill': assistant_prefill,
        'temperature': temperature,
        'analysis': analysis,
        'classification': classification,
        'classification_response': classification_response
    }

def main():
    st.title("🔍 Deepfake Detection VLM")
    st.markdown("Upload images or videos to detect if they contain AI-generated/deepfake content")

    # Sidebar for endpoint configuration
    with st.sidebar:
        st.header("🔧 Configuration")

        endpoint_url = st.text_input(
            "Endpoint URL",
            value="http://100.64.0.3:8001",
            help="Enter the base URL for your vLLM endpoint (without /v1/models suffix)"
        )

        # Load models button
        if st.button("🔄 Load Models"):
            with st.spinner("Loading available models..."):
                available_models = get_available_models(endpoint_url)
                if available_models:
                    st.session_state.available_models = available_models
                    st.success(f"Loaded {len(available_models)} model(s)")
                else:
                    st.session_state.available_models = []
                    st.error("Failed to load models")

        # Model selection
        if "available_models" in st.session_state and st.session_state.available_models:
            selected_model = st.selectbox(
                "Select Model",
                options=st.session_state.available_models,
                help="Choose the model to use for analysis"
            )
        else:
            st.warning("Click 'Load Models' to see available models")
            selected_model = "openbmb/MiniCPM-V-4_5"  # Default fallback

        st.markdown("---")
        st.markdown("**Current Configuration:**")
        st.write(f"**Endpoint:** {endpoint_url}")
        st.write(f"**Model:** {selected_model}")

    uploaded_files = st.file_uploader(
        "Choose images or videos",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'mp4', 'avi', 'mov', 'mkv'],
        accept_multiple_files=True,
        help="Upload multiple images and/or video files for analysis"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        user_prompt = st.text_area(
            "User Prompt",
            value="Is this image real or AI-generated?",
            height=100,
            help="Customize the prompt sent to the VLM"
        )

    with col2:
        assistant_prefill = st.text_area(
            "Assistant Response Prefill",
            value="Let’s examine the style and the synthesis artifacts: \n\n",
            height=100,
            help="Prefill the assistant's response (optional)"
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            help="Controls randomness: 0 = deterministic, higher = more random"
        )

    if uploaded_files and st.button("🔍 Analyze for Deepfakes", type="primary"):
        # Prepare list to store all file data
        file_data = []

        # Process all files first
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type

            if file_type.startswith('image/'):
                image_data = uploaded_file.read()
                file_data.append({
                    'filename': uploaded_file.name,
                    'image_b64': encode_image_to_base64(image_data),
                    'original_image': Image.open(io.BytesIO(image_data))
                })
            elif file_type.startswith('video/'):
                frames = extract_video_frames(uploaded_file)
                for i, frame in enumerate(frames):
                    file_data.append({
                        'filename': f"{uploaded_file.name}_frame_{i+1}",
                        'image_b64': encode_image_to_base64(frame),
                        'original_image': Image.open(io.BytesIO(frame))
                    })

        if file_data:
            st.subheader("🎯 Analysis Results")

            # Store results for export
            all_results = []

            # Analyze each image individually
            for i, data in enumerate(file_data):
                with st.container():
                    st.markdown(f"### 📄 {data['filename']}")

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.image(data['original_image'], caption=data['filename'], use_container_width=True)

                    with col2:
                        with st.spinner(f"Analyzing {data['filename']}..."):
                            result = analyze_single_image(
                                data['image_b64'],
                                data['filename'],
                                user_prompt,
                                assistant_prefill,
                                temperature,
                                endpoint_url,
                                selected_model
                            )
                            all_results.append(result)

                        # Display classification
                        if result['classification'] == "AI Generated":
                            st.error(f"**Classification: {result['classification']}** 🚨")
                        elif result['classification'] == "Real":
                            st.success(f"**Classification: {result['classification']}** ✅")
                        elif result['classification'] == "Error":
                            st.error(f"**Classification: {result['classification']}** ⚠️")
                        else:
                            st.warning(f"**Classification: {result['classification']}** ❓")

                        # Display analysis
                        st.write("**Analysis:**")
                        st.write(result['analysis'])

                        # Debug section for each image
                        with st.expander(f"🐛 Debug: {data['filename']}", expanded=False):
                            st.text_area(
                                "Analysis Response:",
                                value=result['analysis'],
                                height=150,
                                disabled=True,
                                key=f"analysis_{i}"
                            )
                            st.text_area(
                                "Classification Response:",
                                value=result['classification_response'],
                                height=100,
                                disabled=True,
                                key=f"classification_{i}"
                            )

                    st.divider()

            # Export section
            st.subheader("📊 Export Results")

            col1, col2 = st.columns(2)

            with col1:
                st.info(f"Analyzed {len(file_data)} image(s) total")

            with col2:
                if all_results:
                    export_df = create_export_data(all_results)
                    csv_buffer = io.StringIO()
                    export_df.to_csv(csv_buffer, index=False)

                    st.download_button(
                        label="📥 Download CSV Report",
                        data=csv_buffer.getvalue(),
                        file_name=f"deepfake_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        type="primary"
                    )

    if uploaded_files:
        st.subheader("📁 Uploaded Files Preview")
        cols = st.columns(min(len(uploaded_files), 4))

        for i, uploaded_file in enumerate(uploaded_files):
            with cols[i % 4]:
                if uploaded_file.type.startswith('image/'):
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_container_width=True)
                else:
                    st.video(uploaded_file.read())
                    st.caption(uploaded_file.name)

if __name__ == "__main__":
    main()