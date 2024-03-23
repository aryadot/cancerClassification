
---

# Breast Cancer Prediction with K-Nearest Neighbors

This project implements a K-Nearest Neighbor (KNN) classifier to predict whether a patient has breast cancer. Utilizing the Breast Cancer Wisconsin (Diagnostic) dataset, the model is trained to distinguish between malignant and benign tumor characteristics based on features extracted from digitized images of breast mass.

## Project Overview

The goal of this project is to demonstrate the application of a K-Nearest Neighbor classifier on a real-world health dataset. Through this project, users will learn how to preprocess data, split datasets into training and testing sets, train a KNN classifier, and evaluate its performance to predict breast cancer diagnosis.

## Features

- Utilization of the Breast Cancer Wisconsin (Diagnostic) dataset from sklearn.
- Data preprocessing and analysis.
- Training and testing with the K-Nearest Neighbor algorithm.
- Evaluation of the classifier's performance with different values of K.
- Visualization of the classifier's accuracy.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher installed.
- The following Python packages installed: `numpy`, `pandas`, `matplotlib`, `sklearn`.

You can install the required packages using pip:

```
pip install numpy pandas matplotlib scikit-learn
```

## Installation and Setup

1. Clone the repository or download the source code to your local machine.
2. Navigate to the project directory.
3. Ensure you have the prerequisites installed.
4. Run the script with Python:

```
python breast_cancer_classifier.py
```

## Usage

The main script `breast_cancer_classifier.py` is executed from the command line. Upon execution, the script will:

1. Load the dataset.
2. Preprocess and split the data into training and testing sets.
3. Train the KNN classifier with the training data.
4. Evaluate and display the classifier's performance on the testing set.

You can modify the script to experiment with different values of K or to test the classifier with custom input data.

## Contributing

Contributions to this project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name - aboruah@umass.edu

