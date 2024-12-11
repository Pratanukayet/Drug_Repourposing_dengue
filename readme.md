Drug Repurposing for Dengue using FCNN
This repository contains a deep learning pipeline for drug repurposing targeting the dengue virus. The project leverages advanced embeddings for drug and protein representations, combined with a Fully Connected Neural Network (FCNN), to predict dissociation constant (Kd) values for drug-target interactions.

Overview
Drug repurposing accelerates drug discovery by identifying novel uses for existing drugs. This project integrates diverse molecular and biological data to predict drug-protein interactions, aiming to uncover potential therapeutic options for dengue.

Highlights
State-of-the-Art Representations:
Drugs: Processed using ChemBERTa embeddings (transformer-based SMILES processing) and Morgan fingerprints (molecular substructure).
Proteins: Represented using ProtT5 embeddings, capturing sequence-level biological features.
GPU-Accelerated Training:
The pipeline is designed for efficient model training using CUDA-enabled devices.
Explainability:
Model outputs are supported with detailed analysis to interpret predictions.
Pipeline Description
1. Environment Setup
Import essential libraries, including PyTorch, RDKit, transformers, and tape.
2. Embedding Generation
ChemBERTa: SMILES strings are tokenized and processed to extract drug embeddings.
Morgan Fingerprints: Generate numerical fingerprints for molecular analysis.
ProtT5: Protein sequences are transformed into embeddings.
3. Data Preparation
Split the dataset into training (70%), validation (15%), and test (15%).
Normalize Kd values using a log transformation for numerical stability.
4. Model Training
Train an FCNN with attention-based fusion of drug and protein embeddings.
Use dropout and learning rate scheduling to prevent overfitting.
5. Evaluation
Evaluate model performance on validation and test datasets.
Metrics include MSE, RMSE, and R², with results visualized for better interpretability.
6. Predictions
Process new drug-protein pairs to predict Kd values using the trained model.

Repository Structure
├── data/
│   ├── ChemBERT_drug_embeddings.csv
│   ├── drug_morgan_fingerprints.csv
│   ├── protein_prostt5_embeddings.csv
│   ├── train_dataset.csv
│   ├── val_dataset.csv
│   ├── test_dataset.csv
├── models/
│   ├── FCNN_model.pth
├── results/
│   ├── evaluation_metrics.csv
│   ├── prediction_results.csv
├── notebooks/
│   ├── Dengue_Drug_Repurposing_FCNN_Model.ipynb

Prerequisites
Dependencies
Python >= 3.8
PyTorch >= 1.11
RDKit
transformers
tape
numpy
pandas
tqdm
seaborn
matplotlib
scikit-learn
Hardware
A CUDA-enabled GPU is highly recommended for efficient training.


Setup and Usage
Clone the Repository

git clone https://github.com/Pratanukayet/Drug_Repourposing_dengue.git
cd Drug_Repourposing_dengue


Acknowledgments
ChemBERTa: Transformer-based drug embeddings.
ProtT5: Advanced protein embeddings.
RDKit: Molecular structure analysis tools