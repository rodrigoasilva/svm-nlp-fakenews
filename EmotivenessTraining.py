import glob,os
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm

feature = []
label = []
vectorizer = CountVectorizer()

os.chdir("Fake.br-Corpus-master/full_texts/fake-meta-information")
for file in glob.glob("*.txt"):
    f = open(file, encoding="utf8")
    for i, line in enumerate(f):
            if i == 23:
                var = f.read()
                feature.append(var)     #coloca no array "feature" o valor de emotiveness
                label.append("Fake")        #coloca no array "label" se a noticia é fake ou verdadeira


os.chdir("../true-meta-information")
for file in glob.glob("*.txt"):
    f = open(file, encoding="utf8")
    for i, line in enumerate(f):
            if i == 23:
                var = f.read()
                feature.append(var)     #coloca no array "feature" o valor de emotiveness
                label.append("True")        #coloca no array "label" se a noticia é fake ou verdadeira

c = list(zip(feature, label))

random.shuffle(c)       #embaralha um array para ficar devidamente separado as noticias fakes das true

feature, label = zip(*c)


X = vectorizer.fit_transform(feature)       #X vai ser um vetor com a contagem de quantas vezes as palavras do bag of word se repete pra cada texto

model = svm.SVC()
model.fit(X,label)        #treinando o modelo

cv_results = cross_validate(model, X, label)    #porcentagem de acertos do modelo
print(cv_results['test_score'])         #Array com as porcentagens de acertos do cross validation
