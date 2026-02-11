# F1 score for the algorithms
from sklearn.metrics import f1_score

f1_tree = f1_score(y_test, y_pred2)
f1_rf = f1_score(y_test, y_pred3)
f1_svm = f1_score(y_test, y_pred4)
f1_logreg = f1_score(y_test, y_pred)

print("🔹 F1 Score - Decision Tree:", f1_tree)
print("🔹 F1 Score - Random Forest:", f1_rf)
print("🔹 F1 Score - SVM:", f1_svm)
print("🔹 F1 Score - Logistic Regression:", f1_logreg)
