# Initialize Project

Set up the ML/AI project environment locally.

## 1. Create Virtual Environment

```bash
# Using Python venv:
python -m venv .venv

# Or using conda:
conda create -n ai4si python=3.10
conda activate ai4si
```

## 2. Activate Virtual Environment

```bash
# For venv:
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# For conda:
conda activate ai4si
```

## 3. Install Dependencies

```bash
# Install all project dependencies
pip install -r requirements.txt

# Or if using uv:
uv pip install -r requirements.txt
```

## 4. Verify Installation

```bash
# Check Python version (should be 3.8+)
python --version

# Verify key packages are installed
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

## 5. Set Up Data Directories

```bash
# Create necessary data directories
mkdir -p data/raw data/processed data/splits
mkdir -p models/checkpoints
mkdir -p experiments/runs
```

## 6. Run Initial Validation

```bash
# Run tests to verify setup
pytest tests/ -v

# Check code quality
black --check src/ tests/
ruff check src/ tests/
```

## Notes

<!-- Add project-specific notes about initialization here -->
- Ensure you have sufficient disk space for datasets
- GPU setup (if using): Verify CUDA/ROCm installation for PyTorch
- Data access: Ensure you have access to required datasets or download them
