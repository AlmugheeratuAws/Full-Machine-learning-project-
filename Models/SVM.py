# SVC algorithm for classifcation
svm_model=SVC()
svm_model.fit(x_train,y_train)

y_pred4=svm_model.predict(x_test)
print(y_pred4)

# Evaluate the model
SVM_acc_score=accuracy_score(y_test, y_pred4)
SVM_conf_matrix=confusion_matrix(y_test, y_pred4)
SVM_class_report=classification_report(y_test, y_pred4)
print("Accuracy Score: ", SVM_acc_score)
print("Confusion Matrix: ", SVM_conf_matrix)
print("Classification Report: ", SVM_class_report)
