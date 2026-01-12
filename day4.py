# from sklearn.datasets import load_breast_cancer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# X, y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)
# clf = LogisticRegression(max_iter=10000, random_state=0)
# clf.fit(X_train, y_train)
# acc = accuracy_score(y_test, clf.predict(X_test)) * 100
# print(f"Logistic Regression model accuracy: {acc:.2f}%")


# from sklearn.model_selection import train_test_split
# from sklearn import datasets, linear_model, metrics
# digits = datasets.load_digits()
# X = digits.data
# y = digits.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
# reg = linear_model.LogisticRegression(max_iter=10000, random_state=0)
# reg.fit(X_train, y_train)
# y_pred = reg.predict(X_test)
# print(f"Logistic Regression model accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import sklearn
# import warnings
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import KNNImputer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import f1_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import cross_val_score
# warnings.filterwarnings('ignore')
# df= pd.read_csv('Position_Salaries.csv')
# print(df)
# df.info()
# X = df.iloc[:,1:2].values
# y = df.iloc[:,2].values

# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
# x_numerical = df.select_dtypes(exclude=['object']).values
# x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values
# regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
# regressor.fit(x, y)

# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# x_categorical = df.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
# x_numerical = df.select_dtypes(exclude=['object']).values
# x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values
# regressor = RandomForestRegressor(n_estimators=50, random_state=30, oob_score=True)
# regressor.fit(x, y)


# from sklearn.metrics import mean_squared_error, r2_score
# oob_score = regressor.oob_score_
# print(f'Out-of-Bag Score: {oob_score}')
# predictions = regressor.predict(x)
# mse = mean_squared_error(y, predictions)
# print(f'Mean Squared Error: {mse}')
# r2 = r2_score(y, predictions)
# print(f'R-squared: {r2}')

# import numpy as np

# X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.01) # Only the first feature
# X_grid = X_grid.reshape(-1, 1)
# X_grid = np.hstack((X_grid, np.zeros((X_grid.shape[0], 2)))) # Pad with zeros
# plt.scatter(X[:, 0], y, color='blue', label="Actual Data")
# plt.plot(X_grid[:, 0], regressor.predict(X_grid), color='green', label="Random Forest Prediction")
# plt.title("Random Forest Regression Results")
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.legend()
# plt.show()

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# tree_to_plot = regressor.estimators_[0]
# plt.figure(figsize=(20, 10))
# plot_tree(tree_to_plot, feature_names=df.columns.tolist(), filled=True, rounded=True, fontsize=10)
# plt.title("Decision Tree from Random Forest")
# plt.show()

# from sklearn.datasets import load_breast_cancer
# import matplotlib.pyplot as plt
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.svm import SVC
# cancer = load_breast_cancer()
# X = cancer.data[:, :2]
# y = cancer.target
# svm = SVC(kernel="linear", C=1)
# svm.fit(X, y)
# DecisionBoundaryDisplay.from_estimator(
#  svm,
#  X,
#  response_method="predict",
#  alpha=0.8,
#  cmap="Pastel1",
#  xlabel=cancer.feature_names[1],
#  ylabel=cancer.feature_names[2],
#  )
# plt.scatter(X[:, 1], X[:, 1],
#  c=y,
#  s=20, edgecolors="r")
# plt.show()

# from sklearn.datasets import load_breast_cancer
# import matplotlib.pyplot as plt
# from sklearn.inspection import DecisionBoundaryDisplay
# from sklearn.svm import SVC
# cancer = load_breast_cancer()
# X = cancer.data[:, :2]
# y = cancer.target
# svm = SVC(kernel="linear", C=1)
# svm.fit(X, y)
# DecisionBoundaryDisplay.from_estimator(
#  svm,
#  X,
#  response_method="predict",
#  alpha=0.8,
#  cmap="Pastel1",
#  xlabel=cancer.feature_names[0],
#  ylabel=cancer.feature_names[1],
#  )
# plt.scatter(X[:, 0], X[:, 1],
#  c=y,
#  s=20, edgecolors="r")
# plt.title("Doted picture", fontweight='bold', color='Black')
# plt.show()



# import math
# import random
# import pandas as pd
# import numpy as np

# def encode_class(mydata):
#  classes = []
#  for i in range(len(mydata)):
#  if mydata[i][-1] not in classes:
#  classes.append(mydata[i][-1])
#  for i in range(len(classes)):
#  for j in range(len(mydata)):
#  if mydata[j][-1] == classes[i]:
#  mydata[j][-1] = i
#  return mydata

# def splitting(mydata, ratio):
#  train_num = int(len(mydata) * ratio)
#  train = []
#  test = list(mydata)

#  while len(train) < train_num:
#  index = random.randrange(len(test))
#  train.append(test.pop(index))
#  return train, test


# def groupUnderClass(mydata):
#  data_dict = {}
#  for i in range(len(mydata)):
#  if mydata[i][-1] not in data_dict:
#  data_dict[mydata[i][-1]] = []
#  data_dict[mydata[i][-1]].append(mydata[i])
#  return data_dict

# def MeanAndStdDev(numbers):
#  avg = np.mean(numbers)
#  stddev = np.std(numbers)
#  return avg, stddev
# def MeanAndStdDevForClass(mydata):
#  info = {}
#  data_dict = groupUnderClass(mydata)
#  for classValue, instances in data_dict.items():
#  info[classValue] = [MeanAndStdDev(attribute) for attribute in zip(*instances)]
#  return info

# def calculateGaussianProbability(x, mean, stdev):
#  epsilon = 1e-10
#  expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
#  return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo
# def calculateClassProbabilities(info, test):
#  probabilities = {}
#  for classValue, classSummaries in info.items():
#  probabilities[classValue] = 1
#  for i in range(len(classSummaries)):
#  mean, std_dev = classSummaries[i]
#  x = test[i]
#  probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
#  return  probabilities

# def predict(info, test):
#  probabilities = calculateClassProbabilities(info, test)
#  bestLabel = max(probabilities, key=probabilities.get)
#  return bestLabel
# def getPredictions(info, test):
#  predictions = [predict(info, instance) for instance in test]
#  return predictions


# filename = '/content/diabetes_data.csv'
# df = pd.read_csv(filename, header=None, comment='#')
# mydata = df.values.tolist()
# mydata = encode_class(mydata)
# for i in range(len(mydata)):
#  for j in range(len(mydata[i]) - 1):
#  mydata[i][j] = float(mydata[i][j])

#  ratio = 0.7
# train_data, test_data = splitting(mydata, ratio)
# print('Total number of examples:', len(mydata))
# print('Training examples:', len(train_data))
# print('Test examples:', len(test_data))

# info = MeanAndStdDevForClass(train_data)
# predictions = getPredictions(info, test_data)
# accuracy = accuracy_rate(test_data, predictions)
# print('Accuracy of the model:', accuracy)

# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# y_true = [row[-1] for row in test_data]
# y_pred = predictions
# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap='Blues')

# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_score, recall_score, f1_score
# actual = [0, 1, 1, 0, 1, 0, 1, 1]
# predicted = [0, 1, 0, 0, 1, 0, 1, 0]
# precision = precision_score(actual, predicted)
# recall = recall_score(actual, predicted)
# f1 = f1_score(actual, predicted)
# metrics = ['Precision', 'Recall', 'F1 Score']
# values = [precision, recall, f1]
# plt.figure(figsize=(6, 4))
# plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
# plt.ylim(0, 1)
# plt.title('Precision, Recall, and F1 Score')
# plt.ylabel('Score')
# for i, v in enumerate(values):
#  plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

# # plt.show()

# import math
# import random
# import pandas as pd
# import numpy as np

# def encode_class(mydata):
#     classes = []
#     for i in range(len(mydata)):
#         if mydata[i][-1] not in classes:
#             classes.append(mydata[i][-1])
#     for i in range(len(classes)):
#         for j in range(len(mydata)):
#             if mydata[j][-1] == classes[i]:
#                 mydata[j][-1] = i
#     return mydata

# def splitting(mydata, ratio):
#     train_num = int(len(mydata) * ratio)
#     train = []
#     test = list(mydata)
    
#     while len(train) < train_num:
#         index = random.randrange(len(test))
#         train.append(test.pop(index))
#     return train, test

# def groupUnderClass(mydata):
#     data_dict = {}
#     for i in range(len(mydata)):
#         if mydata[i][-1] not in data_dict:
#             data_dict[mydata[i][-1]] = []
#         data_dict[mydata[i][-1]].append(mydata[i])
#     return data_dict

# def MeanAndStdDev(numbers):
#     avg = np.mean(numbers)
#     stddev = np.std(numbers)
#     return avg, stddev

# def MeanAndStdDevForClass(mydata):
#     info = {}
#     data_dict = groupUnderClass(mydata)
#     for classValue, instances in data_dict.items():
#         info[classValue] = [MeanAndStdDev(attribute) for attribute in zip(*instances)]
#     return info

# def calculateGaussianProbability(x, mean, stdev):
#     epsilon = 1e-10
#     expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
#     return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo

# def calculateClassProbabilities(info, test):
#     probabilities = {}
#     for classValue, classSummaries in info.items():
#         probabilities[classValue] = 1
#         for i in range(len(classSummaries)):
#             mean, std_dev = classSummaries[i]
#             x = test[i]
#             probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
#     return probabilities

# def predict(info, test):
#     probabilities = calculateClassProbabilities(info, test)
#     bestLabel = max(probabilities, key=probabilities.get)
#     return bestLabel

# def getPredictions(info, test):
#     predictions = [predict(info, instance) for instance in test]
#     return predictions

# def accuracy_rate(test, predictions):
#     correct = sum(1 for i in range(len(test)) if test[i][-1] == predictions[i])
#     return (correct / float(len(test))) * 100.0

# filename = 'diabetes.csv'

# # Read with headers
# df = pd.read_csv(filename)

# # Convert to list of lists
# mydata = df.values.tolist()

# # Encode class labels
# mydata = encode_class(mydata)

# # Convert features to float
# for i in range(len(mydata)):
#     for j in range(len(mydata[i]) - 1):  # skip the last column (class)
#         mydata[i][j] = float(mydata[i][j])

# ratio = 0.7
# train_data, test_data = splitting(mydata, ratio)

# print('Total number of examples:', len(mydata))
# print('Training examples:', len(train_data))
# print('Test examples:', len(test_data))

# info = MeanAndStdDevForClass(train_data)

# predictions = getPredictions(info, test_data)
# accuracy = accuracy_rate(test_data, predictions)
# print('Accuracy of the model:', accuracy)

# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# y_true = [row[-1] for row in test_data]
# y_pred = predictions

# cm = confusion_matrix(y_true, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot(cmap='Blues')

# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_score, recall_score, f1_score

# actual = [0, 1, 1, 0, 1, 0, 1, 1]
# predicted = [0, 1, 0, 0, 1, 0, 1, 0]

# precision = precision_score(actual, predicted)
# recall = recall_score(actual, predicted)
# f1 = f1_score(actual, predicted)

# metrics = ['Precision', 'Recall', 'F1 Score']
# values = [precision, recall, f1]

# plt.figure(figsize=(6, 4))
# plt.bar(metrics, values, color=['red', 'pink', 'on'])
# plt.ylim(0, 1)
# plt.title('Precision, Recall, and F1 Score')
# plt.ylabel('Score')
# for i, v in enumerate(values):
#     plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')
    
# plt.show()

import numpy as np
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
        distances.sort(key=lambda x: x[0])
        k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [4, 5]
k = 3

prediction = knn_predict(training_data, training_labels, test_point, k)
print(prediction)