# Salmonella Host Colonization Range Classifier

## Overview
This repository contains a machine learning project focused on classifying the host colonization range for various Salmonella strains. 
The project is based on the data from the following paper from Seif et al: https://doi.org/10.1038/s41467-018-06112-5. 

The project is structured into two primary components:

1. **Training Pipeline**: Automates the process of model training, including cross-validation, hyperparameter tuning, and model evaluation.
2. **Model Serving**: Facilitates model deployment through a Flask-based API, containerized using Docker.

The system is designed to be highly configurable, adaptable to various model architectures, and easy to deploy for practical use.

This project corresponds to my Capstone project 1 of the ml-zoomcamp course.

## Getting Started

### Installation
1. Clone the repository:
```
git clone https://github.com/your-username/salmonella-classification.git
```

2. Install required dependencies:
```
pipenv install
```
### Model serving

1. Build the docker image:
```
docker build -t host_range_classifier .
```
2. Run the container:
```
docker run --rm -p 9696:9696 host_range_classifier
```
When succesfully running you should see the following message:
```
Model decision_tree_HostRangeClassifier_v1.bin loaded successfully. Ready to recieve requests and make predictions.
```
3. Test the service:
```
python predict-test.py
```

### Training models
The project contains an automated pipeline for training different models.
All parameters for running the pipeline can be configured by changing the config.py file

1. Change to the trainin_pipeline directory
```
cd training_pipeline
```
2. Specify the paramaters for the pipeline in 'config.py' (see below for description)
3. Run the training script:
```
python train_host-range_classifier.py
```
This will automatically handle cross-validation, hyperparameter tuning and will select the best parameters. The model will be saved at the 'trained_models' directory.

#### Config file explanation (training pipeline)

- `data_path`: Path to data used for model training.
- `model_type`: The type of model to train. Options are 'decision_tree', 'random_forest', and 'logistic_regression'.
- `feature_selection_method*`: Method for selecting features. Options are 'variance' or 'mutual_information'.
- `threshold*`: Threshold for selecting features based on the specified method.
- `metric*`: `roc_auc`
- `model_version`: Version of the final trained model. Choose the name you prefer.
- `save_model`: Option to save the final model with pickle.
- `output_path`: Path to the directory for the saved model.
- `hyperparam_grid`: Dictionary of parameters to test in the hyperparameter tuning and cross-validation step.

\* Parameters marked with '*' are still in development and might not work properly. For now, please use the default ones.
