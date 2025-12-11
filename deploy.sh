#!/bin/bash
# Deepfake Detection Application - Deployment Script

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed"

    # Set the docker compose command
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE="docker-compose"
    else
        DOCKER_COMPOSE="docker compose"
    fi
}

# Build the application
build() {
    echo ""
    echo "Building Docker image..."
    $DOCKER_COMPOSE build
    print_success "Build completed"
}

# Start the application
start() {
    echo ""
    echo "Starting application..."
    $DOCKER_COMPOSE up -d
    print_success "Application started"
    echo ""
    print_info "Access the application at: http://localhost:8501"
    echo ""
    echo "To view logs, run: ./deploy.sh logs"
}

# Stop the application
stop() {
    echo ""
    echo "Stopping application..."
    $DOCKER_COMPOSE down
    print_success "Application stopped"
}

# Restart the application
restart() {
    echo ""
    echo "Restarting application..."
    $DOCKER_COMPOSE restart
    print_success "Application restarted"
}

# View logs
logs() {
    $DOCKER_COMPOSE logs -f
}

# Show status
status() {
    echo ""
    echo "Container status:"
    $DOCKER_COMPOSE ps
    echo ""
    echo "Resource usage:"
    docker stats deepfake-detector-app --no-stream
}

# Clean up
clean() {
    echo ""
    read -p "This will remove all containers and images. Continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleaning up..."
        $DOCKER_COMPOSE down -v
        docker rmi deepfake-detection-deepfake-detector 2>/dev/null || true
        print_success "Cleanup completed"
    else
        print_info "Cleanup cancelled"
    fi
}

# Test connection to model servers
test_connections() {
    echo ""
    echo "Testing connections to model servers..."
    echo ""

    # Define model endpoints
    declare -A endpoints=(
        ["InternVL 2.5"]="http://100.64.0.1:8000/v1/models"
        ["InternVL 3.5"]="http://localhost:1234/v1/models"
        ["MiniCPM-V"]="http://100.64.0.3:8001/v1/models"
        ["Qwen3 VL"]="http://100.64.0.3:8006/v1/models"
    )

    for model in "${!endpoints[@]}"; do
        endpoint="${endpoints[$model]}"
        if curl -s --connect-timeout 3 "$endpoint" > /dev/null 2>&1; then
            print_success "$model is reachable at $endpoint"
        else
            print_warning "$model is NOT reachable at $endpoint"
        fi
    done
    echo ""
    print_info "Note: Make sure model servers are running before using the application"
}

# Show help
show_help() {
    cat << EOF
Deepfake Detection Application - Deployment Script

Usage: ./deploy.sh [command]

Commands:
  build       Build the Docker image
  start       Start the application (builds if needed)
  stop        Stop the application
  restart     Restart the application
  logs        View application logs (follow mode)
  status      Show container status and resource usage
  test        Test connections to model servers
  clean       Remove containers and images
  help        Show this help message

Examples:
  ./deploy.sh build           # Build the image
  ./deploy.sh start           # Start the application
  ./deploy.sh logs            # View logs
  ./deploy.sh test            # Test model server connections

For more information, see README_DOCKER.md
EOF
}

# Main script
main() {
    case "${1:-help}" in
        build)
            check_prerequisites
            build
            ;;
        start)
            check_prerequisites
            build
            start
            ;;
        stop)
            stop
            ;;
        restart)
            restart
            ;;
        logs)
            logs
            ;;
        status)
            status
            ;;
        test)
            test_connections
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
