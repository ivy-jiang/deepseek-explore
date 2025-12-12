# DeepSeek 1.3B Exploration

This project explores the DeepSeek 1.3B model (specifically `deepseek-coder-1.3b-instruct`) and demonstrates how to integrate it with PyTorch for finance applications.

## Setup

1.  Create a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install torch transformers accelerate huggingface_hub
    ```

## Scripts

-   `explore_model.py`: Downloads the model and inspects its architecture and weights.
-   `finance_demo.py`: Demonstrates a custom PyTorch module using DeepSeek for financial headline classification.

## License

[MIT](LICENSE)
