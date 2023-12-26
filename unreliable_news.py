import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Setting display options for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Downloading NLTK stopwords
nltk.download('stopwords')

# Loading the dataset
df = pd.read_csv("train.csv")

# Displaying the first few rows of the dataset
print(df.head())

# Checking for null values in the dataset
print(df.isnull().sum())

# Replacing null values with an empty string
df = df.fillna('')

# Merging the author name and news title into a single column 'content'
df['content'] = df['author'] + ' ' + df['title']

# Separating the data and label
X = df.drop(columns='label', axis=1)
Y = df['label']

# Function for text preprocessing: stemming and removing stopwords
def stemming(content):
    # Removing non-alphabetic characters and lowercasing
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    # Stemming and removing stopwords
    stemmed_content = [PorterStemmer().stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Applying the text preprocessing function
df['content'] = df['content'].apply(stemming)

# Vectorizing the text data using TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(df['content'])
X = vectorizer.transform(df['content'])

# Splitting the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)

# Creating and training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Evaluating the model on the training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data: ', training_data_accuracy)

# Evaluating the model on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data: ', test_data_accuracy)

# Making a prediction on a single instance from the test set
X_new = X_test[2]
prediction = model.predict(X_new)
print('The news is ' + ('Real' if prediction[0] == 0 else 'Fake'))

# Displaying the actual label for the instance
print('Actual label: ', 'Real' if Y_test[7] == 0 else 'Fake')
