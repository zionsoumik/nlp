import numpy as np
from gensim import models
import pandas
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer
from gensim import matutils
#from gensim.models.ldamodel import LdaModel
#from sklearn import cross_validation
from sklearn import dummy
from sklearn.linear_model import LogisticRegression
from sklearn import feature_extraction
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn import naive_bayes
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm
from sklearn.decomposition import LatentDirichletAllocation as LDA
def indent(lines, amount, ch=' '):
    padding = amount * ch
    return padding + ('\n'+padding).join(lines.split('\n'))
models = [svm.LinearSVC(),LogisticRegression()]

model_names = ["Linear SVM","LogisticRegression"]
   #ac=[0,0,0,0,0,0,0,0]


results = {}

def main():
    data1 = pandas.read_csv(r"liwc_input.csv")
    data2 = pandas.read_csv(r'liwc_test.csv')
    trainX = data1.iloc[:, 1:99]
    yTrain = data1.iloc[:, 99]
    testX = data2.iloc[:, 1:99]
    yTest = data2.iloc[:, 99]
    runBaseline = True
    #trainX, testX, yTrain, yTest = cross_validation.train_test_split(X, Y, test_size=0.1, random_state=0)  #test train split

    vectorizer = feature_extraction.text.TfidfVectorizer()
    sentiment_scaler = preprocessing.StandardScaler()
    liwc_scaler=preprocessing.StandardScaler()
    unigrams = vectorizer.fit_transform(trainX["text"]).toarray()
    vectorizer1 = feature_extraction.text.TfidfVectorizer()
    #synst=vectorizer1.fit_transform(trainX["synset"].values.astype('U')).toarray()
    tf_vectorizer =feature_extraction.text.CountVectorizer()
    tf = tf_vectorizer.fit_transform(trainX["text"]).toarray()
    tf_feature_names = tf_vectorizer.get_feature_names()
    lda = LDA(n_topics=10, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
    lda_train = lda.transform(tf)
    sentiment = sentiment_scaler.fit_transform(trainX.ix[:, "pscore":"obscore"])
    liwc=liwc_scaler.fit_transform(trainX.ix[:, "WC":"OtherP"])
    allf = np.hstack((unigrams, lda_train,liwc,sentiment))



    unigrams_t = vectorizer.transform(testX["text"]).toarray()
    liwc_t=liwc_scaler.fit_transform(testX.ix[:, "WC":"OtherP"])
    tf_t = tf_vectorizer.transform(testX["text"]).toarray()
    lda_test = lda.transform(tf_t)
    sentiment_t = sentiment_scaler.transform(testX.ix[:, "pscore":"obscore"])
    #3synst_t = vectorizer1.transform(testX["synset"].values.astype('U')).toarray()
    allf_t = np.hstack((unigrams_t,lda_test,liwc_t,sentiment_t))

    features = {"sentiment":(sentiment,sentiment_t),"lda":(lda_train,lda_test),'unigrams':(unigrams,unigrams_t), "liwc":(liwc,liwc_t),"all":(allf,allf_t)}

    for f in features:
        xTrain = features[f][0]
        xTest = features[f][1]

        if runBaseline:
            baseline = dummy.DummyClassifier(strategy='most_frequent', random_state=0)
            baseline.fit(xTrain, yTrain)
            predictions = baseline.predict(xTest)

            print(indent("Baseline: ", 4))
            print(indent("Test Accuracy: ", 4), metrics.accuracy_score(yTest, predictions))
            print(indent(metrics.classification_report(yTest, predictions), 4))
            print()
            runBaseline = False

        print(indent("Features: ", 4), f)
        count=0
        ac = [0, 0, 0, 0, 0, 0, 0, 0]
        for model, name in zip(models, model_names):
            model.fit(xTrain, yTrain)
            # Simple SVM
            # print('fitting...')
            prediction = model.predict(xTest)
            # Print Accuracy
            print(model)
            print(indent("Test Accuracy: ", 4), metrics.accuracy_score(yTest, prediction))
            print(indent(metrics.classification_report(yTest, prediction), 4))
            print()
            # clf = SVC(C=20.0, gamma=0.00001)
            # clf.fit(X_train, y_train)
            # acc = clf.score(X_test, y_test)

        print()
    print()


print()

if __name__ == '__main__':
    main()