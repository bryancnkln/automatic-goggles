# DeepSeek-Agent Quick Start Guide

Get DeepSeek-Agent running in Docker within minutes!

## Prerequisites

- Docker Desktop 4.10+ ([Download](https://www.docker.com/products/docker-desktop))
- 16GB+ RAM recommended
- 50GB+ free disk space

## For Apple Silicon (M1/M2/M3/M4) Users

```bash
cd DeepSeek-OCR

# Make quick-start script executable
chmod +x scripts/docker-quickstart.sh

# Run interactive setup
./scripts/docker-quickstart.sh

# Select option 3 for Apple Silicon optimization
```

Or manually:

```bash
# Build with ARM64 support
docker-compose build

# Start with Apple optimizations
docker-compose -f docker-compose.yml -f docker-compose.apple.yml up -d

# View logs
docker-compose logs -f
```

## For Linux / CUDA-Enabled Systems

```bash
cd DeepSeek-OCR

# Make quick-start script executable
chmod +x scripts/docker-quickstart.sh

# Run interactive setup
./scripts/docker-quickstart.sh

# Select option 2 for multi-agent training
```

Or manually:

```bash
# Build image
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f agent-trainer-1
```

## For macOS (Intel)

```bash
cd DeepSeek-OCR

# Build
docker-compose build

# Start training
docker-compose up -d agent-trainer-1

# Monitor
docker-compose logs -f agent-trainer-1
```

## Quick Commands

### Start Everything
```bash
# All agents and monitoring
docker-compose up -d

# With Apple optimizations
docker-compose -f docker-compose.yml -f docker-compose.apple.yml up -d
```

### Start Specific Services
```bash
# Just the main trainer
docker-compose up -d agent-trainer-1

# With Jupyter notebook
docker-compose up -d jupyter agent-trainer-1
```

### Monitor Training
```bash
# Live logs from all services
docker-compose logs -f

# Specific service
docker-compose logs -f agent-trainer-1

# Monitoring dashboard
open http://localhost:9000
```

### Interactive Access
```bash
# Open shell in running container
docker-compose exec agent-trainer-1 bash

# Run Python commands
docker-compose exec agent-trainer-1 python -c "import torch; print(torch.__version__)"

# View files
docker-compose exec agent-trainer-1 ls -la /app
```

### Check Status
```bash
# All containers
docker-compose ps

# Resource usage
docker stats

# Container details
docker inspect deepseek-agent:latest
```

### Stop Services
```bash
# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop specific service
docker-compose stop agent-trainer-1

# Remove all Docker images
docker rmi deepseek-agent:latest
```

## Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Jupyter | `http://localhost:8888` | Interactive notebooks |
| Monitoring | `http://localhost:9000` | Training metrics dashboard |
| TensorBoard | `http://localhost:6006` | Training visualization |

## Configuration

### Edit Config
```bash
# Default config in container
/app/config.yaml

# Override locally
./DeepSeek-Agent/config.yaml

# Environment variables
# Copy and edit .env (in project root)
cp DeepSeek-Agent/.env.example .env
```

### Key Settings
- **Device**: Set `DEVICE=mps` for Apple Silicon, `cuda` for NVIDIA
- **Batch Size**: Reduce if out of memory: `BATCH_SIZE=16`
- **Memory**: Adjust `MAX_MEMORY_GB` based on your RAM
- **Workers**: Set `NUM_WORKERS=4` for CPU thread count

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size in config
BATCH_SIZE=8
OMP_NUM_THREADS=2
```

### Container Won't Start
```bash
# Check logs
docker-compose logs agent-trainer-1

# Rebuild from scratch
docker-compose down
docker image rm deepseek-agent:latest
docker-compose build --no-cache
```

### Slow on Apple Silicon
```bash
# Ensure MPS is enabled
docker-compose logs agent-trainer-1 | grep -i mps

# Use Apple-optimized compose file
docker-compose -f docker-compose.yml -f docker-compose.apple.yml up -d
```

### Port Already in Use
```bash
# Change port in docker-compose.yml
# Or kill process using port:
lsof -i :8888
kill -9 <PID>
```

## Development Workflow

### Edit Code & Auto-Reload
```bash
# Modify files locally
vim DeepSeek-Agent/src/training/train_projector.py

# Changes auto-reflected in container (source code mounted)
docker-compose exec agent-trainer-1 python -m pdb scripts/run_agent.py
```

### Run Custom Scripts
```bash
docker-compose exec agent-trainer-1 python \
  src/training/train_projector.py \
  --config custom_config.yaml \
  --epochs 5
```

### Install Additional Packages
```bash
docker-compose exec agent-trainer-1 pip install package-name
```

## Performance Benchmarks (M4 Pro)

| Operation | Time |
|-----------|------|
| Vision projection | ~50ms |
| LLM inference | 1-3 sec |
| Memory retrieval | <10ms |
| Full agent step | 2-5 sec |

## Multi-Agent Training

### Scale to 4 Trainers
```bash
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up -d
```

### Monitor All Agents
```bash
# Dashboard shows aggregate metrics
open http://localhost:9000

# Individual logs
docker-compose logs agent-trainer-1 &
docker-compose logs agent-trainer-2 &
docker-compose logs agent-trainer-3 &
```

## Advanced

### Custom Dockerfile
```bash
# Create Dockerfile.custom extending our image
cat > Dockerfile.custom << 'EOF'
FROM deepseek-agent:latest
RUN pip install your-package
EOF

# Build and test
docker build -f Dockerfile.custom -t deepseek-agent:custom .
```

### Push to Registry
```bash
docker tag deepseek-agent:latest your-registry/deepseek-agent:latest
docker push your-registry/deepseek-agent:latest
```

### Backup Data
```bash
# Save checkpoints
docker run --rm -v deepseek_checkpoints:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/checkpoints.tar.gz -C /data .
```

## Getting Help

1. **Check logs**: `docker-compose logs -f`
2. **Detailed setup**: See `DOCKER_SETUP.md`
3. **Original docs**: Check `DeepSeek-Agent/README.md`
4. **Issues**: Open GitHub issue with `docker-compose ps` output

## Next Steps

1. ✅ Docker running?
   - [ ] Verify: `docker ps`

2. ✅ Agent trained?
   - [ ] Check logs: `docker-compose logs agent-trainer-1 | grep "epoch"`

3. ✅ Want to develop?
   - [ ] Open Jupyter: `http://localhost:8888`
   - [ ] Edit code and it auto-reloads

4. ✅ Ready for production?
   - [ ] See `DOCKER_SETUP.md` > Advanced Deployment

---

**Happy training!** 🚀

For detailed troubleshooting and advanced topics, see [DOCKER_SETUP.md](DOCKER_SETUP.md)
