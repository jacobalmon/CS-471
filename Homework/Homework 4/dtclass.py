from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    iris = load_iris()  # Load Iris Dataset.
    x = iris.data  # Features.
    y = iris.target  # Labels.

    # Split Data into Training and Testing Datasets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=29)

    # Range of Values.
    max_depth_vals = range(1, 15)
    min_samples_leaf_vals = range(1, 11)

    # Creating Lists for the Outcomes of the Hyperparameters.
    max_depth_outcomes = []
    min_samples_leaf_outcomes = []

    # Evaluating Decision Tree Classifier for Max Depth using Cross-Validation.
    for max_depth in max_depth_vals:
        dt_classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=29)
        scores = cross_val_score(dt_classifier, x_train, y_train, cv=5)  # 5-fold cross-validation
        max_depth_outcomes.append(scores.mean())

    # Evaluating Decision Tree Classifier for Min Samples Leaf using Cross-Validation.
    for min_samples_leaf in min_samples_leaf_vals:
        dt_classifier = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=29)
        scores = cross_val_score(dt_classifier, x_train, y_train, cv=5)  # 5-fold cross-validation
        min_samples_leaf_outcomes.append(scores.mean())

    plt.figure(figsize=(12, 6))

    # Plotting the Cross-Validation Accuracy vs. Max Depth.
    plt.subplot(1, 2, 1)
    plt.plot(max_depth_vals, max_depth_outcomes, marker='o')
    plt.title('Cross-Validation Accuracy vs. Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Cross-Validation Accuracy')
    plt.xticks(max_depth_vals)
    plt.grid(True)

    # Plotting the Cross-Validation Accuracy vs. Min Samples Leaf.
    plt.subplot(1, 2, 2)
    plt.plot(min_samples_leaf_vals, min_samples_leaf_outcomes, marker='o')
    plt.title('Cross-Validation Accuracy vs. Min Samples Leaf')
    plt.xlabel('Min Samples Leaf')
    plt.ylabel('Cross-Validation Accuracy')
    plt.xticks(min_samples_leaf_vals)
    plt.grid(True)

    # Display Plots onto the Screen.
    plt.tight_layout()
    plt.show()

    # Find the Best Max Depth Outcome.
    best_max_depth = max_depth_vals[max_depth_outcomes.index(max(max_depth_outcomes))]
    # Find the Best Min Samples Leaf Outcome.
    best_min_samples_leaf = min_samples_leaf_vals[min_samples_leaf_outcomes.index(max(min_samples_leaf_outcomes))]

    # Fitting the Model with these new Best Max Outcome & Min Samples Leaf Outcome.
    final_dt_classifier = DecisionTreeClassifier(max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf, random_state=42)
    final_dt_classifier.fit(x_train, y_train)

    # Find the Test Accuracy from our Testing Data.
    y_pred = final_dt_classifier.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Printing Test Data's Accuracy.
    print(f'Test Dataset Accuracy: {test_accuracy * 100:.2f}%')

    # Visualizing the Decision Tree
    plt.figure(figsize=(5, 5))
    plot_tree(final_dt_classifier, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
    plt.title(f'Decision Tree Visualization (Max Depth: {best_max_depth}, Min Samples Leaf: {best_min_samples_leaf})')
    plt.show()
