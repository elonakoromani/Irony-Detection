"""
Irony detection in English Tweets.
Second method without feature extraction
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# reading the training data
df_train_data = pd.read_csv("preprocessed_trainingdata.csv", encoding='utf-8')

# reading the test data
df_test_data = pd.read_csv("preprocessed_testdata.csv", encoding='utf-8')

# separating the target class from the data (from csv file)
df_train_tweets = df_train_data["Tweet Text"]
df_train_labels = df_train_data["index Label"]

df_test_tweets = df_test_data["Tweet Text"]
df_test_labels = df_test_data["index Label"]

cv = CountVectorizer()

x_traincv = cv.fit_transform(df_train_tweets.values.astype(str))

# model = MultinomialNB()
# model = LinearSVC()
# model = tree.DecisionTreeClassifier()
model = LogisticRegression()

# creating the classifier
print(model.fit(x_traincv, df_train_labels))

x_testcv = cv.transform(df_test_tweets.values.astype(str))
predictions = model.predict(x_testcv)

actual_results = np.array(df_test_labels)

correct_predictions = 0
for i in range(len(predictions)):
    if predictions[i] == actual_results[i]:
        correct_predictions = correct_predictions + 1

accuracy = correct_predictions / len(predictions)

print("The calculated accuracy is:")
print(accuracy_score(df_test_labels, predictions))

print("The calculated accuracy is :", accuracy)

print(classification_report(df_test_labels, predictions, target_names=['Ironic', 'Non ironic']))
