#!/bin/bash
# NexInspect Setup Script for Linux/Mac
# Automates installation and configuration

set -e  # Exit on error

echo "============================================"
echo "  NexInspect - Deepfake Ensemble Setup"
echo "  Linux/Mac Installation Script"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "[1/6] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python 3 is not installed${NC}"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓ Found Python ${PYTHON_VERSION}${NC}"

# Create virtual environment
echo ""
echo "[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists, skipping...${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo ""
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo ""
echo "[4/6] Installing Python dependencies..."
echo "This may take several minutes..."
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Check SPAI weights
echo ""
echo "[5/6] Checking SPAI model weights..."
if [ -f "spai/weights/spai.pth" ]; then
    SIZE=$(du -h spai/weights/spai.pth | cut -f1)
    echo -e "${GREEN}✓ SPAI weights found (${SIZE})${NC}"
else
    echo -e "${RED}✗ SPAI weights not found${NC}"
    echo ""
    echo "Please download SPAI weights manually:"
    echo "1. Visit: https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view"
    echo "2. Download spai.pth (~2GB)"
    echo "3. Create directory: mkdir -p spai/weights"
    echo "4. Move file to: spai/weights/spai.pth"
    echo ""
fi

# Create .env file
echo ""
echo "[6/6] Setting up environment configuration..."
if [ -f ".env" ]; then
    echo -e "${YELLOW}.env file already exists, skipping...${NC}"
else
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env file from template${NC}"
        echo "Please edit .env to add your API keys"
    else
        echo -e "${YELLOW}.env.example not found, skipping...${NC}"
    fi
fi

# Final instructions
echo ""
echo "============================================"
echo -e "${GREEN}Setup Complete!${NC}"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Download SPAI weights (if not already done):"
echo "   https://drive.google.com/file/d/1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI/view"
echo ""
echo "2. (Optional) Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "3. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "4. Launch application:"
echo "   streamlit run app.py"
echo ""
echo "5. Open browser to:"
echo "   http://localhost:8501"
echo ""
echo "============================================"
