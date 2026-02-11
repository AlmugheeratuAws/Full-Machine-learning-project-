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
