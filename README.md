Pre-requisite: You may need to create a Weight & Biases account (https://wandb.ai) for proper implementation.
Update: We've extracted all the image extensions from different subfolders that the original dataset directory contained and categorized into simpler malignant & benign sub-folders.

Datasets from: https://www.kaggle.com/datasets/ambarish/breakhis
# Cancer Tissue Fractality Detector

This project implements a CNN model for classifying tissue images as benign or malignant by leveraging both deep image features from a pre-trained ResNet18 and additional mathematical fractal dimensions computed via a box-counting algorithm. A Streamlit-based UI is also provided for interactive evaluation of the fractal dimension on any tissue image. Training metrics are logged using Weights & Biases (wandb) and final accuracy metrics are exported as a JSON file.

## Features

- **CNN with Fractal Dimension Integration:**  
  Combines ResNet18's feature extraction with a computed fractal dimension to enhance classification performance.

- **Custom Dataset & Data Splitting:**  
  Loads images from two folders (`malignant_tissues` and `benign_tissues`) and splits the data into training (70%), validation (15%), and test (15%) sets.

- **Real-Time Logging with wandb:**  
  Tracks training progress, loss, and accuracy metrics in real-time.

- **Streamlit UI for Inference:**  
  Provides an easy-to-use interface where users can upload a tissue image, compute its fractal dimension, and receive an interpretation based on a predefined threshold (∼1.65).

- **JSON Metrics Export:**  
  Saves final training and testing accuracy metrics to `accuracy_metrics.json`.

## File Structure

```
.
├── datasets/
│   ├── malignant_tissues/    # Folder with >2000 images of malignant tissue
│   └── benign_tissues/       # Folder with >2000 images of benign tissue
├── fractal_dimension.py      # Contains the box-counting algorithm to compute fractal dimensions.
├── model.py                  # Defines the CNN model (ResNet18 with fractal dimension integration).
├── train.py                  # Training script with custom dataset, model training, and wandb logging.
├── streamlit_app.py          # Streamlit UI to display fractal dimensions and classification output.
├── requirements.txt          # Python dependencies.
└── README.md                 # This file.
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/cancer-tissue-fractality-detector.git
   cd cancer-tissue-fractality-detector
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model

Ensure your dataset is structured as follows:

```
./datasets/
    ├── malignant_tissues   # Folder with malignant tissue images
    └── benign_tissues      # Folder with benign tissue images
```

To train the model, run:

```bash
python train.py
```

During training:
- The dataset is automatically split into training, validation, and test sets (70-15-15).
- The model logs training progress and metrics using wandb.
- The best performing model on the validation set is saved as `best_model.pth`.
- Accuracy metrics are exported to `accuracy_metrics.json`.

### 2. Running the Streamlit UI

To launch the user interface:

```bash
streamlit run streamlit_app.py
```

The UI allows you to:
- Upload a tissue image.
- Compute and display its fractal dimension.
- Receive an interpretation of whether the tissue might be malignant (fractal dimension > ∼1.65).

> **Note:** The threshold of ∼1.65 is based on current experimental evidence and may require calibration based on your specific dataset.

## Weights & Biases (wandb) Integration

- Before training, create an account on [Weights & Biases](https://wandb.ai/) if you haven't already.
- The `train.py` script logs all training metrics to your wandb dashboard, where you can monitor the performance in real time.

## Troubleshooting

- **Multiprocessing Error on Windows:**  
  If you encounter an error related to starting new processes, make sure that all code launching DataLoader workers is wrapped inside the `if __name__ == "__main__":` guard. This is already implemented in `train.py`.

## License

This project is licensed under the MIT License.

## Acknowledgements

- **PyTorch & torchvision:** for providing state-of-the-art deep learning tools.
- **Weights & Biases:** for seamless experiment tracking and visualization.
- **Research Community:** for insights on fractal dimensions in tissue analysis.

## Contact

For any questions or issues, please open an issue in the repository or contact [info.aamodpaudel@gmail.com].
