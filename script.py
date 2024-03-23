import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load the data
breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])  # First data point
print(breast_cancer_data.feature_names)  # Feature names
print(breast_cancer_data.target)  # Target array
print(breast_cancer_data.target_names)  # Target names

# Checking the target of the first data point
if breast_cancer_data.target[0] == 0:
    print("The first data point is malignant.")
else:
    print("The first data point is benign.")

# Split the data
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)

# Confirming the split
print(len(training_data))
print(len(training_labels))


# Create a KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
classifier.fit(training_data, training_labels)

# Test the classifier
print(classifier.score(validation_data, validation_labels))

accuracies = []
for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

# k values
k_list = range(1, 101)

# Plotting
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()

