# Read the dataset and define variable called na_values to make Python recognize missing data marked with ? as NaN
df=pd.read_csv('adult.csv', na_values='?')
df.head()

#  print the information f the dataset
df.info()

df.shape

# print the number of duplicated values
df.duplicated().sum()

# drop the duplicated values
df.drop_duplicates(inplace=True)
df.duplicated().sum()

df.isnull().sum()

# print the number of the null values as percentage
x= df.isnull().sum()
print(x*100/len(df))

