# Target: income (classification: <=50K or >50K) because it fit better for classification tasks
x=df.drop(['income'], axis=1)
y=df['income']

# splitting the data into test data and training data
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True)
