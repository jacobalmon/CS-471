from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iris = load_iris() # Load Iris Dataset.
    x = iris.data # Features.
    y = iris.target # Labels.

    # Split Data into Training, Validation, and Testing Datasets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=29)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=29)

    max_depth_vals = range(1, 15)
    min_samples_leaf_vals = range(1, 11)

    # Creating Lists for the Outcomes of the Hyper Parameters.
    max_depth_outcomes = []
    min_samples_leaf_outcomes = []

    # Evaluating Decision Tree Classifier for Max Depth.
    for max_depth in max_depth_vals:
      dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=29)
      dt_classifier.fit(x_train, y_train)
      outcomes = dt_classifier.score(x_valid, y_valid)
      max_depth_outcomes.append(outcomes)

     # Evaluating Decision Tree Classifier for Min Samples Leaf.
    for min_samples_leaf in min_samples_leaf_vals:
      dt_classifier = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=29)
      dt_classifier.fit(x_train, y_train)
      outcomes = dt_classifier.score(x_valid, y_valid)
      min_samples_leaf_outcomes.append(outcomes)

    plt.figure(figsize=(12,6))

    # Plotting the Validation Accuracy vs. Max Depth.
    plt.subplot(1,2,1)
    plt.plot(max_depth_vals, max_depth_outcomes, marker='o')
    plt.title('Validation Accuracy vs. Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Validation Accuracy')
    plt.xticks(max_depth_vals)
    plt.grid(True)

    # Plotting the Validation Accuracy vs. Min Samples Leaf.
    plt.subplot(1,2,2)
    plt.plot(min_samples_leaf_vals, min_samples_leaf_outcomes, marker='o')
    plt.title('Validation Accuracy vs. Min Samples Leafs')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Validation Accuracy')
    plt.xticks(min_samples_leaf_vals)
    plt.grid(True)

    # Display Plots onto the Screen.
    plt.tight_layout()
    plt.show()

    # Find the Best Max Depth Outcome.
    best_max_depth = max_depth_vals[max_depth_outcomes.index(max(max_depth_outcomes))]
    # Find the Best Min Samples Lead Outcome.
    best_min_samples_leaf = min_samples_leaf_vals[min_samples_leaf_outcomes.index(max(min_samples_leaf_outcomes))]

    # Fitting the Model with these new Best Max Outcome & Min Samples Leaf Outcome.
    final_dt_classifier = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf, random_state=42)
    final_dt_classifier.fit(x_train, y_train)

    # Find the Test Accuracy from our Testing Data.
    y_pred = final_dt_classifier.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f'Test Dataset Accuracy: {test_accuracy * 100}')
