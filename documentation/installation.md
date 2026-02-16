# Installation

This guide will walk you through setting up the **Spilled Energy** library.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Git**: For cloning the repository.
- **Python**: Version 3.10 or higher is recommended.
- **[uv](https://github.com/astral-sh/uv)**: A fast Python package installer and resolver. We strongly recommend using `uv` for managing dependencies in this project.

### Install `uv` (if not already installed)
If you don't have `uv`, you can install it via pip:
```bash
pip install uv
```
Or view their [official installation guide](https://github.com/astral-sh/uv#installation) for other methods.

## Setup Steps

1.  **Clone the Repository**
    Start by cloning the `spilled-energy` repository to your local machine:
    ```bash
    git clone https://github.com/OmnAI-Lab/spilled-energy.git
    cd spilled-energy
    ```

2.  **Install Dependencies**
    Use `uv` to sync the project dependencies. This will create a virtual environment (defaulting to `.venv`) and install all required packages (PyTorch, Transformers, Datasets, etc.):
    ```bash
    uv sync
    ```

3.  **Activate the Virtual Environment**
    We recommended activating the environment to use the installed packages directly:
    ```bash
    source .venv/bin/activate
    ```

4.  **Verify Installation**
    You can verify that everything is set up correctly by running the test script:
    ```bash
    uv run src/scripts/test_measure_exact_answer.py
    ```

## Environment Variables
If you plan to use models or datasets that require authentication (e.g., Llama 3 from Hugging Face), you may need to set up your environment variables.

Copy the example environment file:
```bash
cp .env.example .env
```
Then edit `.env` and add your Hugging Face token:
```bash
HF_TOKEN=your_huggingface_token_here
```
Alternatively, you can log in via the Hugging Face CLI:
```bash
huggingface-cli login
```
