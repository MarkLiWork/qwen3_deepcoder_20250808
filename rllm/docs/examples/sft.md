# Math SFT Training Example

This example demonstrates supervised fine-tuning (SFT) of math reasoning models using the RLLM framework. The SFT training pipeline generates high-quality trajectories from a teacher model and fine-tunes a student model on the successful trajectories.

Our examples use the following:
* Qwen/Qwen2.5-Math-1.5B as the base model
* agentica-org/DeepScaleR-1.5B-Preview as the teacher model for trajectory generation
* DeepScaleR math dataset for training data

## Overview

The Math SFT examples demonstrate:

- How to generate high-quality training data from teacher model trajectories
- How to perform supervised fine-tuning on successful math reasoning trajectories
- How to fine-tune math reasoning models using the DeepScaleR dataset

## Quick Start

### Model Hosting

Start a vLLM server with OpenAI-compatible API for the teacher model:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model agentica-org/DeepScaleR-1.5B-Preview \
    --host 0.0.0.0 \
    --port 30000 \
    --dtype bfloat16
```

The server should be accessible at `http://localhost:30000/v1`

### Dataset Preparation

First prepare the dataset:

```bash
cd examples/sft
python prepare_math_data.py
```

Generate SFT training data from teacher model trajectories:

```bash
cd examples/sft
python generate_sft_data.py --num_samples 1000 --reward_threshold 1.0 --output large_sft_data.parquet
```

This will:
- Load problems from the DeepScaleR math dataset  
- Generate trajectories using the teacher model
- Filter trajectories by reward threshold
- Save successful trajectories in SFT format

#### Configuration Options

- `--num_samples`: Number of problems to generate trajectories for (default: 100)
- `--reward_threshold`: Minimum reward score to include trajectory (default: 0.0)
- `--output`: Output file name (default: "sft_data.parquet")

### Training

Run SFT with the generated data:

```bash
bash examples/sft/train_math_sft.sh
```

#### Training Configuration Options

You can modify the training script parameters:

- `model.partial_pretrain`: Base model to fine-tune
- `trainer.total_epochs`: Number of training epochs
- `data.train_batch_size`: Total batch size across all GPUs
- `data.micro_batch_size_per_gpu`: Batch size per GPU
- `data.max_length`: Maximum sequence length
- `data.train_files`: Training data file
- `data.val_files`: Validation data file

The training script will:
1. Load the base model (Qwen2.5-Math-1.5B)
2. Fine-tune on the generated SFT data
3. Save checkpoints to `outputs/qwen2.5_math_sft/`

### Evaluation

Evaluate the trained model using the saved checkpoint:

```bash
cd examples/sft
python run_sft_model.py --model_path outputs/qwen2.5_math_sft/
```

Replace `outputs/qwen2.5_math_sft/` with the actual path to your trained model checkpoint.

## Code Reference

### SFT Data Generator

Main script for generating SFT training data:

```python title="examples/sft/generate_sft_data.py"
--8<-- "examples/sft/generate_sft_data.py"
```

### Math SFT Model Evaluator

Script for evaluating SFT model performance:

```python title="examples/sft/run_sft_model.py"
--8<-- "examples/sft/run_sft_model.py"
```

For detailed setup instructions, see the [README](https://github.com/agentica-project/rllm-internal/blob/v0.1/examples/sft/README.md) in the sft example directory.
