# Deepfake Detection - Quick Start Guide

Get up and running with the containerized Deepfake Detection application in 5 minutes!

## 🚀 Quick Start (3 Steps)

### Step 1: Ensure Model Servers are Running

Make sure your vLLM model servers are running and accessible:

```bash
# Test connectivity to model servers
./deploy.sh test
```

You should see the models you have running (InternVL, MiniCPM-V, Qwen3 VL).

**Note:** If model servers are not accessible, update the URLs in `config.py` to match your setup.

### Step 2: Start the Application

```bash
# Build and start the application
./deploy.sh start
```

This will:
- Build the Docker image (first time only, ~2-3 minutes)
- Start the Streamlit application
- Make it available at http://localhost:8501

### Step 3: Access the Application

Open your browser and navigate to:

**http://localhost:8501**

You should see the Deepfake Detection interface with two tabs:
- 🔍 **Detection** - Analyze individual images/videos
- 📊 **Evaluation** - Batch evaluation with metrics

---

## 📋 Common Commands

```bash
# View application logs
./deploy.sh logs

# Check status
./deploy.sh status

# Restart application
./deploy.sh restart

# Stop application
./deploy.sh stop

# Full cleanup (remove containers and images)
./deploy.sh clean
```

---

## 🔧 Configuration

### Model Server URLs

The application connects to model servers defined in `config.py`:

- **InternVL 2.5 8B**: Port 8000
- **InternVL 3.5 8B**: Port 1234
- **MiniCPM-V 4.5**: Port 8001
- **Qwen3 VL 32B**: Port 8006

**If these URLs don't work for you:**

1. Open `config.py`
2. Update the `base_url` values for each model
3. Restart: `./deploy.sh restart`

### Changing the Port

If port 8501 is already in use:

1. Edit `docker-compose.yml`
2. Change `"8501:8501"` to `"YOUR_PORT:8501"`
3. Restart: `./deploy.sh start`

---

## 📁 Data Files

The application uses these directories (automatically mounted):

- `results/` - Evaluation results
- `testing_files/` - Test images and ground truth
- `misc/` - Miscellaneous data
- `analysis_output/` - Generated reports

All data persists even when the container stops.

---

## 🔍 Troubleshooting

### Can't connect to model servers?

```bash
# Check if models are reachable
./deploy.sh test

# If not reachable, update config.py with correct URLs
# For Docker host machine (Mac/Windows):
"base_url": "http://host.docker.internal:PORT/v1/"

# For external IP:
"base_url": "http://YOUR_IP:PORT/v1/"
```

### Port already in use?

```bash
# Check what's using port 8501
sudo lsof -i :8501

# Or change the port in docker-compose.yml
ports:
  - "8502:8501"  # Use port 8502 instead
```

### Application not starting?

```bash
# View detailed logs
./deploy.sh logs

# Check container status
./deploy.sh status

# Rebuild from scratch
./deploy.sh clean
./deploy.sh start
```

---

## 📖 Usage Examples

### 1. Single Image Detection

1. Go to **🔍 Detection** tab
2. Select a model from the dropdown
3. Upload an image or video
4. Wait for automatic analysis
5. Ask follow-up questions in the chat

### 2. Batch Evaluation

1. Go to **📊 Evaluation** tab
2. Upload multiple test images
3. Upload ground truth CSV (columns: `filename`, `label`)
4. Select number of runs per image (for consensus)
5. Choose which models to evaluate
6. Click **🚀 Run Evaluation**
7. Download results as Excel file

### 3. Generate Analysis Reports

From inside the container or host:

```bash
# Generate comprehensive analysis report
python3 generate_report_updated.py
```

Reports will be saved in `analysis_output/` with visualizations and detailed metrics.

---

## 🛠️ Development Mode

To develop with live code reloading:

1. Edit `docker-compose.yml` and add:
   ```yaml
   volumes:
     - ./app.py:/app/app.py
     - ./config.py:/app/config.py
     - ./shared_functions.py:/app/shared_functions.py
   ```

2. Restart: `./deploy.sh restart`
3. Edit code on your host machine
4. Streamlit will auto-reload changes

---

## 📚 Next Steps

- Read [README_DOCKER.md](README_DOCKER.md) for detailed Docker documentation
- Check [COMPREHENSIVE_REPORT.md](analysis_output/COMPREHENSIVE_REPORT.md) for model performance analysis
- Review [config.py](config.py) to understand model configurations

---

## 🆘 Need Help?

1. Check logs: `./deploy.sh logs`
2. Test model connections: `./deploy.sh test`
3. View status: `./deploy.sh status`
4. See full documentation: [README_DOCKER.md](README_DOCKER.md)

---

**Happy Deepfake Detecting! 🕵️‍♂️**
