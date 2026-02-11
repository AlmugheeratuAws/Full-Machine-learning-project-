# RandomForest algorithm for classification
Random_model=RandomForestClassifier()
Random_model.fit(x_train, y_train)

y_pred3=Random_model.predict(x_test)
print(y_pred3)

# Evaluate the model
Random_acc_score=accuracy_score(y_test, y_pred3)
Random_conf_matrix=confusion_matrix(y_test, y_pred3)
Random_class_report=classification_report(y_test, y_pred3)
print("Accuracy Score: ", Random_acc_score)
print("Confusion Matrix: ", Random_conf_matrix)
print("Classification Report: ", Random_class_report)

# feature importance for RandomForest algorithm
importances = Random_model.feature_importances_
feature_names = x_train.columns

importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importances_df.head(10))
plt.title('Top 10 Important Features (Random Forest)')
plt.show()
