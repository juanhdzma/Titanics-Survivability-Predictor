# Survivability Predictor - ANN, LR, SVM

The Survivability Predictor project aims to predict the survivability of individuals using machine learning models. The project utilizes a dataset that contains various features related to individuals, such as age, sex, number of siblings/spouses aboard, number of parents/children aboard, fare, and more. These features are used to train the models and make predictions about whether an individual survived or not.

## Dataset

The dataset used in this project contains information about passengers aboard the Titanic. It consists of two CSV files: `train.csv` and `test.csv`. The `train.csv` file is used for training and evaluating the models, while the `test.csv` file is used for final predictions.

The dataset includes the following features:

- `PassengerId`: Unique identifier for each passenger
- `Survived`: Whether the passenger survived (0 = No, 1 = Yes)
- `Pclass`: Ticket class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
- `Name`: Passenger's name
- `Sex`: Passenger's sex (Male or Female)
- `Age`: Passenger's age in years
- `SibSp`: Number of siblings/spouses aboard the Titanic
- `Parch`: Number of parents/children aboard the Titanic
- `Ticket`: Ticket number
- `Fare`: Passenger fare
- `Cabin`: Cabin number
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

Please ensure that you download the dataset from [Kaggle](https://www.kaggle.com/code/juanhdzma/survivability-predictor-ann-lr-svm-0-772) and place the `train.csv` and `test.csv` files in the appropriate location before running the code.

## Models

The three machine learning models used in this project are:

1. Artificial Neural Network (ANN): A neural network model that can learn complex patterns in the data. It consists of multiple layers of interconnected nodes (neurons) and is trained using backpropagation.

2. Logistic Regression (LR): A statistical model used to predict binary outcomes. It estimates the probability of an event occurring based on the input features.

3. Support Vector Machine (SVM): A supervised learning algorithm that can be used for both classification and regression tasks. It finds a hyperplane that separates the classes with the maximum margin.

By comparing the performance of these models, the project aims to determine which model provides the most accurate predictions for the survivability of individuals.

## Requirements

To run this project, you need to have the following dependencies installed:

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Keras
- TensorFlow

You can install these dependencies by running the following command:

```bash
pip install pandas numpy scikit-learn keras tensorflow
