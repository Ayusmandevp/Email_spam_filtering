import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#dataset
df = pd.read_csv("spam.csv")

X = df["text"]
y = df["label"]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

#dataset splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Training
model = MultinomialNB()
model.fit(X_train, y_train)

#input
message = input("Enter a message to check if it is spam or not: ")

#Convert message to vector
message_vector = vectorizer.transform([message])

#Predicting
prediction = model.predict(message_vector)

print("Result:", prediction[0])