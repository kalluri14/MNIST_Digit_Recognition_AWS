README.txt
----------

Project Overview:
-----------------
This project implements a real-time digit recognition tool using the MNIST dataset, leveraging AWS SageMaker for efficient model training, hyperparameter tuning, and deployment. The main focus is to build a scalable and robust solution utilizing cloud-based services, allowing real-time interaction with the model through a simple web application interface developed using Streamlit.

Project Workflow:
-----------------
- Setting up AWS SageMaker environment and IAM roles for necessary permissions.
- Preparing and preprocessing data using SageMaker Data Wrangler to ensure data quality.
- Training a CNN model on SageMaker using a custom-built algorithm tailored for MNIST digit recognition.
- Performing hyperparameter tuning to enhance model accuracy and efficiency.
- Deploying the model with an endpoint for real-time predictions.
- Developing a Streamlit-based web application to provide a user-friendly interface for interacting with the model.

Folder Structure:
-----------------
1. model_training.ipynb - Jupyter Notebook containing:
   - Execution of training jobs and hyperparameter tuning jobs.
   - Endpoint deployment for serving predictions.
   - A simple Streamlit UI for interacting with the trained model.
   
   *Code Ownership Note:* The main structure and code logic for training and tuning, as well as Streamlit UI integration, were developed by us. Any specific methods for hyperparameter tuning or deployment that follow standard SageMaker documentation might be inspired by official AWS resources.

2. mnist_train.py - Python script containing:
   - Model definition, including architecture, training, and evaluation metrics.
   - Calculation and display of training loss and validation accuracy.
   
   *Code Ownership Note:* The model structure, training loop, and evaluation metrics were written by me. Portions of the script related to specific utility functions or baseline model implementations have been adapted from public examples documentation (e.g., initial MNIST classification examples).

3. inference.py - Python script containing:
   - Loading of the trained model from a `gzip.tar` file stored on S3.
   - Extraction and deserialization logic for model loading and inference.
   
   *Code Ownership Note:* The code for extracting and loading the trained model from S3 is customized for this project but leverages some functionality provided by AWS SDKs and SageMaker utilities.

Dependencies:
-------------
Ensure that all dependencies listed in `model_training.ipynb` are installed before executing the code. The primary dependencies include AWS SDK for Python (boto3), SageMaker, TensorFlow/PyTorch (as applicable), and Streamlit for UI interactions.

Simple Web Interface:
-------------
Streamlit UI is used for developing simple interface to interact with model to show predictions, there are 3 files

1. Home.py - Home page 
2. pages/ Image upload.py - has the code logic to load the best model , run inference and give predictions when a image is uploaded as input
3. pages/ Draw a Digit.py - has the code logic to load the best model , run  inference and give predictions upon drawing digit on canavas

Instructions for Use:
---------------------
1. Run the `model_training.ipynb` to train, tune hyperparameters, and deploy the model.
2. `mnist_train.py` serves as the entry point for training. Modify this file to experiment with different model architectures or evaluation metrics.
3. Use `inference.py` to load and interact with the trained model for inference.
4. To execute th streamlit web interface, run the command streamlit run Home.py

Key Challenges and Solutions:
-----------------------------
- Encountered PyTorch version compatibility issues with SageMaker notebook instances; resolved by adapting supported versions i.e 1.9.1.
- Managed cost-effectiveness of cloud resources by using spot instances, reducing training costs significantly.
- Local training took 5 minutes, while SageMaker training took 13 minutes on an m4.xlarge instance; optimized resources accordingly.

Project Contributors:
---------------------
- Narasimha Royal: SageMaker environment setup, data preprocessing using Data Wrangler, and creation of Streamlit UI.
- Divija Kalluri: Model training, hyperparameter tuning, deployment, and endpoint creation on SageMaker.

Note:
-----
*This project is executed in AWS Free Tier.*
