# fill the null values with mode
df['workclass']=df['workclass'].fillna(df['workclass'].mode()[0])
df['occupation']=df['occupation'].fillna(df['occupation'].mode()[0])
df['native.country']=df['native.country'].fillna(df['native.country'].mode()[0])
df.head()

# print the shape of the dataset before encoding
print("Shape before encoding:", df.shape)

# encode the data useing OneHotEncoding because they are nominal categrical data
encoded_data = ['workclass', 'occupation', 'native.country', 'marital.status', 'relationship', 'race']
df = pd.get_dummies(df, columns=encoded_data, drop_first=True)

# print the shape of the data after encoding
print("Shape after encoding:", df.shape)

# encode the data with LabelEncoding because the are ordinal categorical data
le=LabelEncoder()
df['sex']=le.fit_transform(df['sex'])
df['income']=le.fit_transform(df['income'])

# encode the data using OrdinalEncoder because it is ordinal categorical data and fit better for this category
ce=OrdinalEncoder()
df['education']=ce.fit_transform(df[['education']])

# Scaling the data using MinMaxScaler, I didn't Scale the income column beacaues it is a discrete value
scaler=MinMaxScaler()
columns_for_scaling=['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
df[columns_for_scaling]=scaler.fit_transform(df[columns_for_scaling])
df.head()

