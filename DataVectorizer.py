import glob,os
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm

bag = []
label = []
vectorizer = CountVectorizer(min_df = 0.05)  #esse atributo min_df é que ele so seleciona as palavras que aparecem em pelo menos 5% dos textos

os.chdir("Fake.br-Corpus-master/size_normalized_texts/fake")
for file in glob.glob("*.txt"):
    f = open(file, encoding="utf8")
    var = f.read()
    bag.append(var)     #coloca no array "bag", o texto inteiro das noticias fake
    label.append("Fake")        #coloca no array "label" se a noticia é fake ou verdadeira

os.chdir("../true")
for file in glob.glob("*.txt"):
    f = open(file, encoding="utf8")
    var = f.read()
    bag.append(var)     #coloca no array "bag", o texto inteiro das noticias verdadeiras
    label.append("True")        #coloca no array "label" se a noticia é fake ou verdadeira


c = list(zip(bag, label))

random.shuffle(c)       #embaralha um array para ficar devidamente separado as noticias fakes das true

bag, label = zip(*c)

x_train,y_train = bag[0:len(bag)],label[0:len(label)]       #separa em traning set e validation set
x_test,y_test = bag[len(bag):],label[len(label):]

X = vectorizer.fit_transform(x_train)       #X vai ser um vetor com a contagem de quantas vezes as palavras do bag of word se repete pra cada texto

tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}       #função para achar os melhores parametros para aprimorar a porcentagem de acerto do modelo

model = GridSearchCV(svm.SVC(), tuned_parameters)
model.fit(X,y_train)        #treinando o modelo

print(model.score(cv.transform(x_test),y_test))         #porcentagem de acertos do modelo
