# MATH5472: Final project
# spaVAE: Dependency-aware Deep Generative Models for Spatial Omics Data

spaVAE is a dependency-aware deep generative spatial variational autoencoder designed to probabilistically model count data while accounting for spatial correlations. By combining Gaussian process (GP) priors with traditional Gaussian priors, spaVAE explicitly captures spatial dependencies, making it an efficient and robust tool for analyzing spatially resolved transcriptomics (SRT) data.

## Key Features
- **Adaptive Spatial Dependency Modeling**: Learns spatial correlations directly from data using Gaussian process priors
- **Hybrid Embedding Approach**: Combines GP and Gaussian embeddings to capture both spatially dependent and independent variations
- **Negative Binomial Reconstruction Loss**: Effectively addresses over-dispersion and library size variations in count data
- **Computational Efficiency**: Leverages sparse Gaussian process regression for scalability to large datasets
- **Multitasking Capability**: Supports tasks like dimensionality reduction, clustering, denoising, differential expression analysis, spatial interpolation, resolution enhancement, and identifying spatially variable genes

## Repository Structure
The repository is organized as follows:
- `examples/`: Contains raw and preprocessed datasets (Human DLPFC samples: 151673, 151510, 151507), configurations for training/testing, and utility scripts.
- `spaVAE/`:
  - `dataset/`: Handles raw data preprocessing and dataset creation.
  - `model/`: Defines the kernel functions, encoder/decoder architectures, sparse GP, and the spaVAE model.
  - `trainer/`: Manages training and validation, logging, and checkpointing.
  - `utils/`: Provides helper functions for setup, visualization, and metrics calculation.

## Getting Started

### Prerequisites
- Python >= 3.9
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
  ```

### Training the model
To train spaVAE, prepare a configuration file for your model (`config.json`) and a configuration file for your logger (such as `logger-config.json`) defining the model and dataset settings, and run:
```bash
python train.py -c <your-config-file.json> -l <your-logger-config-file.json>
```

After training, results will be stored in the `experiments/` directory, including:
- Model checkpoints (`model_best.pth`, `checkpoint-epoch<N>.pth`)
- Logs (`log.log`) and TensorBoard files for visualization

### Testing the model
For testing, use the command:
```bash
python test_Human_DLPFC_sample_<sample-id>.py \
  -c "examples/Human_DLPFC/sample_<sample-id>/config.json" \
  -l <your-logger-config-file.json>
```

Testing results (e.g., logs (`log.log`) and TensorBoard files for visualization, pngs of latent space visualizations, denoised counts) will be saved in the `tests/` directory.

### Visualization with TensorBoard
To visualize training or testing logs with TensorBoard run:
```bash
tensorboard --logdir=<path-to-log-dir/>
```

## More info
For a detailed explanation of the methodology, implementation, and results, please refer to the `MATH5472-Final-project-report.pdf` included in this repository.
