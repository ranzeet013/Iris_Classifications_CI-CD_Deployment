import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import os
import numpy as np

def main():
    """
    Main function to train multiple models on the Iris dataset,
    log the models using MLflow, and track their performance metrics.
    """
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a new experiment
    experiment_name = "Iris_Classification_Experiment"  # Name your experiment
    mlflow.set_experiment(experiment_name)  # Set the experiment

    # Define the models to be trained
    models = {
        "version_1": LogisticRegression(),
        "version_2": RandomForestClassifier(),
        "version_3": SVC(),
        "version_4": KNeighborsClassifier(),
        "version_5": GaussianNB()
    }

    # Create a directory for saving models if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Set the tracking URI for MLflow
    mlflow.set_tracking_uri("file:./mlruns")  # Use a local directory for tracking MLflow runs

    # Train and save each model with MLflow tracking
    for version, model in models.items():
        with mlflow.start_run() as run:  # Start a new MLflow run
            
            # Train the model on the training data
            model.fit(X_train, y_train)

            # Prepare an input example (first row of the training set) for logging
            input_example = np.array([X_train[0]]).reshape(1, -1)  # Reshape to 2D array
            
            # Log the trained model to MLflow with input_example
            mlflow.sklearn.log_model(model, f"iris_model_{version}", input_example=input_example)

            # Log parameters and metrics for tracking
            mlflow.log_params({
                "version": version,  # Version of the model
                "model_type": model.__class__.__name__,  # Type of the model (e.g., LogisticRegression)
                "train_accuracy": model.score(X_train, y_train),  # Training accuracy
                "test_accuracy": model.score(X_test, y_test)  # Testing accuracy
            })

            # Inform the user that the model has been saved and tracked
            print(f"Model {version} saved and tracked!")

if __name__ == "__main__":
    main()
