# Lightweight Image Classification with PyTorch

This project provides a flexible and robust pipeline for training and evaluating several lightweight image classification models using PyTorch, `timm`, and `MLflow`. It features a 5-fold stratified cross-validation setup to ensure reliable evaluation of model performance.

## Features

- **Four Lightweight Backbones**: Includes `MobileNetV3-Large`, `ViT-Small`, `EfficientFormer-L1`, and `ShuffleNetV2-x1.0`.
- **Generic Classifier**: A wrapper that can adapt any backbone for classification.
- **5-Fold Stratified Cross-Validation**: For robust evaluation.
- **Extensive Experiment Tracking**: Uses `MLflow` to log parameters, metrics, system information, and artifacts.
- **Detailed Metrics**: Calculates and logs accuracy, F1-score, precision, recall, MCC, Cohen's Kappa, and more.
- **Plotting**: Generates and saves confusion matrices, ROC curves, and precision-recall curves.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lewbei/A-Reproducible-Benchmark-of-AdamW-Augmented-Lightweight-Models-for-Trash-Classification.git
    cd A-Reproducible-Benchmark-of-AdamW-Augmented-Lightweight-Models-for-Trash-Classification
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

1.  **Organize your dataset**:
    Your dataset should be in the `ImageFolder` format, where each sub-directory in the root folder represents a class.
    ```
    your_path/
    ├── class_a/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── class_b/
        ├── image3.jpg
        └── image4.jpg
    ```

2.  **Update the dataset path**:
    Open `loop_train.py` and modify the `root_dir` variable to point to your dataset's root directory.
    ```python
    if __name__ == "__main__":
        root_dir = r"your_path"  # <--- CHANGE THIS
        ...
    ```

3.  **Run the training script**:
    ```bash
    python loop_train.py
    ```

## View Results

The script will log all experiments to `MLflow`. You can view the results by running the MLflow UI:
```bash
mlflow ui
```
This will start a local server, typically at `http://127.0.0.1:5000`, where you can compare runs, view metrics, and analyze artifacts.
