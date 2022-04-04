import glob,os
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm


label = []
feature =[]
vectorizer = CountVectorizer()
lines_to_read = [14,16,18,20,21,22,23,24]
os.chdir("Fake.br-Corpus-master/full_texts/fake-meta-information")
for file in glob.glob("*.txt"):
    f = open(file, encoding="utf8")
    var1 = f.read().splitlines()
    var = []
    for i,position in enumerate(var1):
        if i in lines_to_read:
            var.append(position)
    feature.append(var)
    label.append("Fake")        #coloca no array "label" se a noticia é fake ou verdadeira


os.chdir("../true-meta-information")
for file in glob.glob("*.txt"):
    f = open(file, encoding="utf8")
    var1 = f.read().splitlines()
    var = []
    for i,position in enumerate(var1):
        if i in lines_to_read:
            var.append(position)
    feature.append(var)
    label.append("True")        #coloca no array "label" se a noticia é fake ou verdadeira
c = list(zip(feature, label))

random.shuffle(c)       #embaralha um array para ficar devidamente separado as noticias fakes das true

feature, label = zip(*c)


#tuned_parameters = {'gamma': [1e-3, 1e-4],
#                     'C': [1, 10, 100, 1000]}

#model = GridSearchCV(svm.SVC(kernel='linear'), tuned_parameters)
model = svm.SVC(kernel='linear')
model.fit(feature,label)        #treinando o modelo

#y_pred = model.predict(X_test)
#print(model.best_params_)
cv_results = cross_validate(model, feature, label)    #porcentagem de acertos do modelo
print(cv_results['test_score'])         #Array com as porcentagens de acertos do cross validation
