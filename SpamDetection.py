import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pandas as pd
import functions


##training

dataframe = pd.read_csv("spam.csv")

        #dividir o dataset em training set e validation set
x= dataframe["EmailText"]
y=dataframe["Label"]
x_train,y_train = x[0:4457],y[0:4457]
x_test,y_test = x[4457:],y[4457:]

print(x[1])
cv=CountVectorizer()
features = cv.fit_transform(x_train)

        #criar um modelo
tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}

model = GridSearchCV(svm.SVC(), tuned_parameters)
model.fit(features,y_train)

#quais sao os melhores parametros
print(model.best_params_)
#testar a acuracia do modelo
print(model.score(cv.transform(x_test),y_test))

#testar o modelo
f = open('vocab.txt')
email_text = f.read()
email_text = [email_text]
transform  = cv.transform(email_text)
result = model.predict(transform)
print(result)
