 

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# import numpy as np
# from sklearn.datasets import make_classification

# x,y = make_classification(
#     n_samples=1000,n_features=20,n_informative=10,n_classes=2,random_state=42)

# c_space =np.logspace(-5,8,15)
# param_grid={'c':c_space}
# logreg=LogisticRegression()
# logreg_cv=GridSearchCV(logreg,param_grid,cv=5)
# logreg_cv.fit(x,y)
# print("tuned logistic regration parameters:{}".format(logreg_cv.best_params_))
# print("best scope is {}".format(logreg_cv.best_score_))  

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# import numpy as np
# from sklearn.datasets import make_classification

# # Create dataset
# X, y = make_classification(
#     n_samples=1000,
#     n_features=20,
#     n_informative=10,
#     n_classes=2,
#     random_state=42
# )

# # Hyperparameter grid (NOTE: 'C' is uppercase)
# C_space = np.logspace(-5, 8, 15)
# param_grid = {'C': C_space}

# # Logistic Regression model
# logreg = LogisticRegression(max_iter=1000, solver='liblinear')

# # Grid Search
# logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
# logreg_cv.fit(X, y)

# # Results
# print("Tuned Logistic Regression parameters:", logreg_cv.best_params_)
# print("Best score:", logreg_cv.best_score_)

# from sklearn.datasets import make_classification

# x,y=make_classification(n_samples=1000,n_features=20,n_informative=10,n_classes=2,random_state=42)

# from scipy.stats import randint
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import RandomizedSearchCV

# paran_dist={
#     "max_depth":[3,None],
#     "max_features":randint(1,9),
#     "max_sample_leaf":randint(1,9),
#     "criterion":["gini","entropy"]

# }

# tree=DecisionTreeClassifier()
# tree_cv=RandomizedSearchCV(tree,param_dist,cv=5)
# tree_cv.fit(x,y)
# print("Tuned Decision Regresion parameters:{}",format(tree_cv.best_params_))
# print("Best score {}:",formate(tree_cv.best_score_))

# import numpy as np
# from sklearn.datasets import make_classification
# x,y=make_classification(n_samples=1000,
#                         n_features=20,
#                         n_informative=10,
#                         n_classes=2,
#                         random_state=42)
# from scipy.stats import randint
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import RandomizedSearchCV

# param_dist ={
#     "max_depth":[3,None],
#     "max_features":randint(1,9),
#     "min_samples_leaf":randint(1,9),
#     "criterion":["gini","entropy"]
# }
# tree =DecisionTreeClassifier()
# tree_cv=RandomizedSearchCV(tree,param_dist,cv=5)
# tree_cv.fit(x,y)
# print("Tuned decision Regression parameters:{}", format(tree_cv.best_params_))
# print("Best score {}:", format(tree_cv.best_score_))

# # import numpy as np        
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

y_true = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 1, 1]

# Metrics
cm = confusion_matrix(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Output
print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", roc_auc)


# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(fpr,tpr,color='arkorange',lw=2,
#          label='ROC curve(area=%0,2f)'%oc_auc)
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(fpr,tpr,color='darkorange',lw=2,
#          label='ROC curve (area=%0,2f)'%roc_auc)

# plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
# plt.xlim([0.0,1.0])
# plt.ylim([0.0,1.05])

# plt.xlabel('false positive rate')
# plt.ylabel('true positive rate')

# plt.title('receiver operating characteristics')

# plt.legend(loc='lower right')
# plt.show()

# import matplotlib.pyplot as plt

# plt.figure()
# plt.plot(
#     fpr,
#     tpr,
#     color='darkorange',
#     lw=2,
#     label='ROC curve (area=%0.2f)' % roc_auc
# )
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])

# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')

# plt.legend(loc='lower right')
# plt.show()
  

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Step 1: Define data
y_true = [0,1,1,0,1,0,0,1,1,0]
y_pred = [0,1,0,0,1,1,0,1,1,1]

# Step 2: Compute ROC values
fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Step 3: Plot ROC curve
plt.figure()
plt.plot(
    fpr,
    tpr,
    color='darkorange',
    lw=2,
    label='ROC curve (area = %0.2f)' % roc_auc
)

plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')

plt.legend(loc='lower right')
plt.show()