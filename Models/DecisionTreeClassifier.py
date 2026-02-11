# Decisiontree algorithm for classification
Decision_model=DecisionTreeClassifier()
Decision_model.fit(x_train, y_train)

y_pred2=Decision_model.predict(x_test)
print(y_pred2)

# Evaluate the model
Decision_acc_score=accuracy_score(y_test, y_pred2)
Decision_conf_matrix=confusion_matrix(y_test, y_pred2)
Decision_class_report=classification_report(y_test, y_pred2)
print("Accuracy Score: ", Decision_acc_score)
print("Confusion Matrix: ", Decision_conf_matrix)
print("Classification Report: ", Decision_class_report)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Assuming Decision_model is your trained Decision Tree Classifier model
# Assuming x_train are your training features

plt.figure(figsize=(20,10))
plot_tree(Decision_model, feature_names=x_train.columns, filled=True, class_names=['<=50K', '>50K'])
plt.show()
