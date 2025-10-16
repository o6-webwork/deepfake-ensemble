# 🔍 Deepfake Detection VLM

A Streamlit-based web application for detecting AI-generated content using Vision Language Models (VLMs). This tool provides a user-friendly interface for analyzing images and videos to determine if they contain AI-generated or deepfake content.

## Features

### 🎯 Core Functionality
- **Multi-format Support**: Analyze images (PNG, JPG, JPEG, GIF, BMP) and videos (MP4, AVI, MOV, MKV)
- **Individual Analysis**: Each uploaded file gets analyzed separately with dedicated result sections
- **2-Step Classification Process**:
  1. Detailed analysis of visual artifacts and inconsistencies
  2. Binary classification based on the analysis (Real vs AI Generated)
- **Real-time Results**: Live progress indicators and immediate feedback

### 🔧 Configuration
- **Dynamic Endpoint Configuration**: Connect to any vLLM deployment via sidebar
- **Model Selection**: Automatically load and select from available models at the endpoint
- **Temperature Control**: Adjust model creativity/determinism (0.0 - 2.0)
- **Custom Prompts**: Modify analysis prompts and assistant response prefills

### 📊 Data Export
- **CSV Export**: Download comprehensive analysis results including:
  - Filename and metadata
  - Endpoint URL and model name
  - User prompt and assistant prefill
  - Temperature setting
  - Full analysis text
  - Classification result
  - Raw classification response
  - Timestamp

### 🐛 Debug Features
- **Raw Response Viewer**: Expandable sections showing complete VLM outputs
- **Step-by-step Tracking**: Separate debug info for analysis and classification steps
- **Error Handling**: Clear error messages for API failures

## Installation

### Prerequisites
- Python 3.8+
- Access to a vLLM deployment with vision capabilities

### Setup

1. **Clone or download the project files**

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

1. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run deepfake_detector.py
   ```

3. **Open your browser** to the displayed URL (typically `http://localhost:8501`)

### Configuration

1. **Configure Endpoint** (Sidebar):
   - Enter your vLLM endpoint URL (e.g., `http://100.64.0.3:8001`)
   - Click "🔄 Load Models" to fetch available models
   - Select your preferred model from the dropdown

2. **Customize Analysis** (Main Panel):
   - Modify the user prompt to focus on specific detection criteria
   - Adjust the assistant response prefill to guide analysis structure
   - Set temperature for response consistency (0.0 = deterministic)

### Analyzing Content

1. **Upload Files**: Use the file uploader to select images or videos
2. **Review Settings**: Ensure your prompt, prefill, and temperature are configured
3. **Start Analysis**: Click "🔍 Analyze for Deepfakes"
4. **Review Results**: Each file displays:
   - Classification (Real/AI Generated)
   - Detailed analysis explanation
   - Debug information (expandable)

### Exporting Results

- Click "📥 Download CSV Report" in the Export Results section
- CSV includes all analysis data with timestamp for record-keeping

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Direct integration with vLLM OpenAI-compatible API
- **Analysis Pipeline**: Two-step process for improved accuracy
  1. Vision analysis with custom prompts
  2. Text-based classification of the analysis

### API Integration
- Compatible with any vLLM deployment using OpenAI API format
- Automatic endpoint discovery via `/v1/models`
- Support for both image+text and text-only API calls

### Video Processing
- Automatic frame extraction from video files
- Configurable frame sampling (default: 5 frames per video)
- Each frame analyzed as individual image

## Dependencies

```
streamlit>=1.28.0
requests>=2.31.0
Pillow>=10.0.0
opencv-python>=4.8.0
pandas>=2.0.0
```

## Configuration Examples

### Typical vLLM Endpoints
- Local deployment: `http://localhost:8000`
- Remote server: `http://your-server-ip:8001`
- Cloud deployment: `https://your-endpoint.com`

### Recommended Models
- **MiniCPM-V series**: Good balance of speed and accuracy
- **LLaVA models**: Strong vision understanding capabilities
- **Qwen-VL models**: Multilingual support

### Prompt Examples

**Basic Detection**:
```
Is this image real or AI-generated?
```

**Detailed Analysis**:
```
Analyze this image for signs of AI generation. Look for artifacts like:
- Unnatural lighting or shadows
- Inconsistent textures
- Facial anomalies
- Background inconsistencies
```

## Troubleshooting

### Common Issues

**"Failed to load models"**:
- Check that the endpoint URL is correct
- Ensure the vLLM server is running and accessible
- Verify network connectivity

**"API call failed"**:
- Confirm the selected model supports vision tasks
- Check server logs for detailed error information
- Verify the endpoint has sufficient resources

**Slow analysis**:
- Reduce image resolution before upload
- Lower the max_tokens parameter in the code
- Use a smaller/faster model

## Contributing

This tool is designed for defensive security analysis. Contributions should focus on:
- Improved detection accuracy
- Better user experience
- Enhanced export capabilities
- Bug fixes and performance improvements

## License

[Add your license information here]

## Disclaimer

This tool is intended for research, educational, and defensive security purposes. Results should be validated through multiple methods for critical applications. The accuracy depends on the underlying VLM model and training data.