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

. ├── datasets/ │ ├── malignant_tissues/ # Folder with >2000 images of malignant tissue │ └── benign_tissues/ # Folder with >2000 images of benign tissue ├── fractal_dimension.py # Contains the box-counting algorithm to compute fractal dimensions. ├── model.py # Defines the CNN model (ResNet18 with fractal dimension integration). ├── train.py # Training script with custom dataset, model training, and wandb logging. ├── streamlit_app.py # Streamlit UI to display fractal dimensions and classification output. ├── requirements.txt # Python dependencies. └── README.md

