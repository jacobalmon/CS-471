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

    # For Testing Purposes.
    print(f'Training Set Size: {x_train.shape[0]}')
    print(f'Validation Set Size: {x_valid.shape[0]}')
    print(f'Testing Set Size: {x_test.shape[0]}')

    max_depth_vals = range(1, 15)
    min_samples_leaf_vals = range(1, 11)

    max_depth_outcomes = []
    min_samples_leaf_outcomes = []

    for max_depth in max_depth_vals:
      dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=29)
      dt_classifier.fit(x_train, y_train)
      outcomes = dt_classifier.score(x_valid, y_valid)
      max_depth_outcomes.append(outcomes)

    for min_samples_leaf in min_samples_leaf_vals:
      dt_classifier = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=29)
      dt_classifier.fit(x_train, y_train)
      outcomes = dt_classifier.score(x_valid, y_valid)
      min_samples_leaf_outcomes.append(outcomes)

    plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(max_depth_vals, max_depth_outcomes, marker='o')
    plt.title('Validation Accuracy vs. Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Validation Accuracy')
    plt.xticks(max_depth_vals)
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(min_samples_leaf_vals, min_samples_leaf_outcomes, marker='o')
    plt.title('Validation Accuracy vs. Min Sample Leafs')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Validation Accuracy')
    plt.xticks(min_samples_leaf_vals)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    best_max_depth = max_depth_vals[max_depth_outcomes.index(max(max_depth_outcomes))]
    best_min_samples_leaf = min_samples_leaf_vals[min_samples_leaf_outcomes.index(max(min_samples_leaf_outcomes))]

    final_dt_classifier = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf, random_state=42)
    final_dt_classifier.fit(x_train, y_train)

    y_pred = final_dt_classifier.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f'Test Dataset Accuracy: {test_accuracy * 100}')
