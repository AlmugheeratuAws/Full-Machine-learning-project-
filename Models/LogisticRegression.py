# LogisticRegression for classification
logistic_model=LogisticRegression(max_iter=1000)
logistic_model.fit(x_train, y_train)

y_pred=logistic_model.predict(x_test)
print(y_pred)

# Evaluate the model
Logistic_acc_score=accuracy_score(y_test, y_pred)
Logistic_conf_matrix=confusion_matrix(y_test, y_pred)
Logistic_class_report=classification_report(y_test, y_pred)
print("Accuracy Score: ", Logistic_acc_score)
print("Confusion Matrix: ", Logistic_conf_matrix)
print("Classification Report: ", Logistic_class_report)
