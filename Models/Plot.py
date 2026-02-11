# Plot the accuracy_score of each algorithm and compare it
models=['LogisticRegression' ,'DecesionTreeClassifier', 'RandomForestClassifier', "SVC"]
accuracies=[Logistic_acc_score,Decision_acc_score,Random_acc_score, SVM_acc_score]
plt.bar(models,accuracies,color=['blue','green','red','purple','orange'], width=0.3)
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.ylim(0.5,1.0)
plt.show()
