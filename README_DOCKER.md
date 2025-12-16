# Deepfake Detection Application - Docker Deployment Guide

This guide explains how to run the Deepfake Detection application in a Docker container.

## 📋 Prerequisites

- Docker Engine 20.10+ installed
- Docker Compose 2.0+ installed
- Access to vLLM model servers (InternVL, MiniCPM-V, Qwen3 VL)
- At least 4GB RAM available for the container

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the application
docker compose up --build

# Or run in detached mode
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the application
docker compose down
```

The application will be available at: **http://localhost:8501**

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t deepfake-detector .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/testing_files:/app/testing_files \
  -v $(pwd)/misc:/app/misc \
  -v $(pwd)/analysis_output:/app/analysis_output \
  deepfake-detector
```

## 📁 Project Structure

```
deepfake-detection/
├── Dockerfile              # Container definition
├── docker compose.yml      # Multi-container orchestration
├── requirements.txt        # Python dependencies
├── .dockerignore          # Files to exclude from build
├── .env.example           # Environment variables template
├── app.py                 # Main Streamlit application
├── config.py              # Model configurations
├── shared_functions.py    # Shared utility functions
├── generate_report.py     # Report generation script
├── generate_report_updated.py  # Updated report script
├── results/               # Evaluation results (mounted as volume)
├── testing_files/         # Test images (mounted as volume)
├── misc/                  # Miscellaneous files (mounted as volume)
└── analysis_output/       # Generated reports (mounted as volume)
```

## 🔧 Configuration

### Model Server URLs

The application connects to vLLM model servers defined in `config.py`:

- **InternVL 2.5 8B**: `http://100.64.0.1:8000/v1/`
- **InternVL 3.5 8B**: `http://localhost:1234/v1/`
- **MiniCPM-V 4.5**: `http://100.64.0.3:8001/v1/`
- **Qwen3 VL 32B**: `http://100.64.0.3:8006/v1/`

**Important:** These URLs assume the model servers are running on your host machine or accessible network. You may need to update `config.py` based on your setup:

#### If models are on the same Docker network:
```python
# In config.py, change localhost to service name
"base_url": "http://model-service:8000/v1/"
```

#### If models are on the host machine:
```python
# Use host.docker.internal (Mac/Windows) or host network mode (Linux)
"base_url": "http://host.docker.internal:8000/v1/"
```

#### For Linux host network mode:
```bash
# Add to docker compose.yml under deepfake-detector service:
network_mode: "host"
```

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

## 📂 Data Volumes

The following directories are mounted as volumes for persistence:

- `./results` → Evaluation results (Excel files)
- `./testing_files` → Test images and ground truth
- `./misc` → Miscellaneous data files
- `./analysis_output` → Generated analysis reports and visualizations

These volumes ensure your data persists even if the container is stopped or removed.

## 🛠️ Common Tasks

### View Application Logs
```bash
docker compose logs -f deepfake-detector
```

### Restart the Application
```bash
docker compose restart
```

### Rebuild After Code Changes
```bash
docker compose up --build
```

### Access Container Shell
```bash
docker compose exec deepfake-detector /bin/bash
```

### Stop and Remove Everything
```bash
docker compose down
docker compose down -v  # Also remove volumes
```

## 🔍 Troubleshooting

### Issue: Cannot connect to model servers

**Solution 1:** Check if model servers are accessible from the container
```bash
docker compose exec deepfake-detector curl http://100.64.0.1:8000/v1/models
```

**Solution 2:** Update URLs in `config.py` to use appropriate network addressing:
- For host machine: `http://host.docker.internal:PORT/v1/`
- For same Docker network: `http://service-name:PORT/v1/`
- For external IP: `http://EXTERNAL_IP:PORT/v1/`

### Issue: Port 8501 already in use

**Solution:** Change the port mapping in `docker compose.yml`:
```yaml
ports:
  - "8502:8501"  # Change 8502 to any available port
```

### Issue: Out of memory

**Solution:** Increase memory limits in `docker compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 8G  # Increase as needed
```

### Issue: Permission denied on volumes

**Solution:** Fix permissions on host directories:
```bash
sudo chown -R $USER:$USER results/ testing_files/ misc/ analysis_output/
```

## 🏗️ Production Deployment

### Using Docker Compose in Production

Create a `docker compose.prod.yml`:

```yaml
version: '3.8'

services:
  deepfake-detector:
    image: your-registry/deepfake-detector:latest
    restart: always
    ports:
      - "80:8501"
    volumes:
      - /path/to/persistent/results:/app/results
      - /path/to/persistent/testing_files:/app/testing_files
    environment:
      - STREAMLIT_SERVER_PORT=8501
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
```

Deploy:
```bash
docker compose -f docker compose.prod.yml up -d
```

### Using Kubernetes

Example deployment manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: deepfake-detector
spec:
  replicas: 2
  selector:
    matchLabels:
      app: deepfake-detector
  template:
    metadata:
      labels:
        app: deepfake-detector
    spec:
      containers:
      - name: deepfake-detector
        image: your-registry/deepfake-detector:latest
        ports:
        - containerPort: 8501
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
          requests:
            memory: "2Gi"
            cpu: "1"
        volumeMounts:
        - name: data
          mountPath: /app/results
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: deepfake-data
```

## 🔐 Security Considerations

1. **API Keys**: If using external APIs, use environment variables instead of hardcoding
2. **Network Isolation**: Use Docker networks to isolate services
3. **Read-only Filesystem**: Add `read_only: true` to container config where possible
4. **User Permissions**: Run as non-root user (add to Dockerfile):
   ```dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```
5. **Secrets Management**: Use Docker secrets or external secret managers for sensitive data

## 📊 Monitoring

### Health Checks

The container includes a health check that pings Streamlit's health endpoint:

```bash
# Check container health
docker compose ps

# Manual health check
curl http://localhost:8501/_stcore/health
```

### Resource Usage

Monitor container resources:

```bash
# Real-time stats
docker stats deepfake-detector-app

# Detailed info
docker inspect deepfake-detector-app
```

## 🔄 Updates and Maintenance

### Update Application Code

1. Pull latest code changes
2. Rebuild and restart:
```bash
git pull
docker compose up --build -d
```

### Update Dependencies

1. Modify `requirements.txt`
2. Rebuild:
```bash
docker compose build --no-cache
docker compose up -d
```

## 📝 Development Workflow

For active development, use volume mounts for code:

```yaml
# Add to docker compose.yml
services:
  deepfake-detector:
    volumes:
      - ./app.py:/app/app.py
      - ./config.py:/app/config.py
      - ./shared_functions.py:/app/shared_functions.py
```

This allows code changes without rebuilding the container. Streamlit will auto-reload on file changes.

## 🆘 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Docker logs: `docker compose logs -f`
3. Verify model server connectivity
4. Check available resources (CPU, memory, disk)

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [vLLM Documentation](https://docs.vllm.ai/)
