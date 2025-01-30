# GPT-from-scratch

This repository contains various Python scripts and modules for training and fine-tuning a GPT model from scratch.

## Repository Structure

- `src/config.py`: Contains configuration settings for different GPT models.
- `src/data`: Contains scripts for downloading datasets and creating dataloaders.
- `src/finetuning`: Contains scripts for fine-tuning the model using LoRA.
- `src/inference.py`: Used for generating text using the trained model.
- `src/train.py`: Used for training the model.
- `src/training`: Contains utility functions for calculating loss during training.

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model, run the `src/train.py` script:

```bash
python src/train.py
```

### Generating Text

To generate text using the trained model, run the `src/inference.py` script:

```bash
python src/inference.py
```

