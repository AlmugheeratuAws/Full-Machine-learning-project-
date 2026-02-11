# LogisticRegression for classification
logistic_model=LogisticRegression(max_iter=1000)
logistic_model.fit(x_train, y_train)

y_pred=logistic_model.predict(x_test)
print(y_pred)
