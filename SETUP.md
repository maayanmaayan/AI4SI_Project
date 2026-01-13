# Project Setup Guide

This guide will help you set up the development environment for the 15-Minute City Service Gap Prediction System.

## Prerequisites

- Python 3.9+ (recommended: 3.10 or 3.11)
- pip (Python package manager)
- Git

## Initial Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (if needed):
   ```bash
   # Create .env file from template (if .env.example exists)
   # Add any API keys or configuration as needed
   # See PRD.md Section 9 for configuration details
   ```

4. **Configure model settings**:
   ```bash
   cp models/config.yaml.example models/config.yaml
   # Edit models/config.yaml with your preferred hyperparameters
   ```

## Verify Setup

Run a quick check to ensure everything is installed:

```bash
python -c "import torch; import geopandas; import osmnx; print('All core dependencies installed!')"
```

## Project Structure

The project structure follows the architecture defined in `PRD.md`:

- `data/` - Data storage (raw, processed, splits)
- `models/` - Model architecture and configuration
- `src/` - Source code (data collection, training, evaluation)
- `notebooks/` - Jupyter notebooks for exploration
- `tests/` - Test suite
- `experiments/` - Experiment outputs and logs

## Next Steps

1. Review `PRD.md` for project requirements and architecture
2. Review `CURSOR.md` for development conventions and PIV loop guidelines
3. Start with Phase 1: Data Collection & Feature Engineering (see PRD.md Section 12)

## Development Workflow

See `README.md` for the development workflow and `CURSOR.md` for PIV loop conventions.

## Troubleshooting

- **OSMnx installation issues**: May require additional system dependencies (see [OSMnx documentation](https://osmnx.readthedocs.io/))
- **Geopandas installation issues**: May require GDAL system libraries
- **PyTorch installation**: Visit [pytorch.org](https://pytorch.org/) for platform-specific installation instructions
