import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD, PCA
from argparse import Namespace
import os
import pickle
from utils import *
from preprocessing import *




class Classifier(object):
    def __init__(self, args, train_df, processor: TextProcessor=None, classifier=None, labelencoder=None):
        self.args = args
        self.train_df = train_df
        self.texts = self.train_df.text.tolist()
        self.labels = self.train_df.label.tolist()

        if labelencoder is None:
            labelencoder = LabelEncoder().fit(sorted(set(self.labels)))
        self.labelencoder = labelencoder

        vocab = None
        stopwords = None

        if processor is None:
            processor = TextProcessor()
        self.processor = processor

        if args.stopword_file != '':
            stopwords = self.load_stopword(args.stopword_file)
            
        if args.vocab_file != '':
            vocab = self.load_vocab(args.vocab_file)
        self.vectorizer = TfidfVectorizer(vocabulary=vocab, stop_words=stopwords, ngram_range=args.ngram_range, sublinear_tf=args.sublinear_tf)

        if classifier is None:
            if args.kernel == 'linear':
                classifier = svm.LinearSVC(C=args.C, random_state=42)
            else:
                classifier = svm.SVC(C=args.C, kernel=args.kernel, random_state=42)

        self.classifier = classifier

    @classmethod
    def from_csv(cls, args, csv_path, processor: TextProcessor=None, classifier=None):
        data_df = pd.read_csv(csv_path)
        return cls(args, data_df, processor, classifier)

    def train(self):
        self.vectorizer.fit(self.train_df.text.tolist())
        X = self.vectorizer.transform(self.texts)
        print(X.shape)
        self.classifier.fit(X, self.labelencoder.transform(self.labels))

    def predict(self, text):
        text = self.processor.transform(text)
        tfidf = self.vectorizer.transform([text])
        return self.classifier.predict([tfidf])

    def predict_all(self, raw_documents):
        documents = [self.processor.transform(text) for text in raw_documents]
        tfidf = self.vectorizer.transform(documents)
        return self.classifier.predict(tfidf)

    def predict_label(self, raw_documents):
        documents = [self.processor.transform(text) for text in raw_documents]
        tfidf = self.vectorizer.transform(documents)
        preds = self.classifier.predict(tfidf)
        return self.labelencoder.inverse_transform(preds)

    def sorce(self, raw_documents, y_targets):
        documents = [self.processor.transform(text) for text in raw_documents]
        tfidf = self.vectorizer.transform(documents)
        return self.classifier.score(tfidf, y_targets)

    def load_vocab(self, vocab_file):
        try:
            vocab_df = pd.read_csv(vocab_file)
            vocab = vocab_df.vocab.tolist()
            return {word: index for index, word in enumerate(vocab)}
        except FileNotFoundError:
            print('Vocab file is not exist!')
            return None

    def load_stopword(self, stopword_file):
        try:
            with open(stopword_file, 'r') as pf:
                stopwords = pf.readlines()
                stopwords = [word.replace('\n', '') for word in stopwords]
            return stopwords
        except FileNotFoundError:
            print('Stopword file is not exist!')
            return None
    def evaluate(self, test_df_or_testfile, show_cm_matrix=False):
        test_df = None
        if isinstance(test_df_or_testfile, str):
            if os.path.exists(test_df_or_testfile):
                test_df = pd.read_csv(test_df_or_testfile)
        else:
            test_df = test_df_or_testfile

        print("Test size", len(test_df))
        texts = test_df.text.to_list()
        labels = test_df.label.to_list()
        y_labels = self.labelencoder.transform(labels)
        y_pred = self.predict_all(texts)
        print(self.classifier)
        score(y_true=y_labels, y_pred=y_pred)

        classes = self.labelencoder.classes_
        pred_labels = self.labelencoder.inverse_transform(y_pred)
        cm = confusion_matrix(y_true=labels, y_pred=pred_labels, labels=classes)

        plot_confusion_matrix(cm, normalize=False, target_names=classes,
                              title="Confusion Matrix", show=show_cm_matrix)

        plot_confusion_matrix(cm, normalize=True, target_names=classes,
                              title="Confusion Matrix(Normalize)", show=show_cm_matrix)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='data/full_data/data.csv')
    parser.add_argument('--test_file', type=str, default='data/full_data/test.csv')
    parser.add_argument('--vocab_file', type=str, default='vocab/vocab.csv')
    parser.add_argument('--stopword_file', type=str, default='')
    parser.add_argument('--ngram_range', type=tuple, default=(1, 2))
    parser.add_argument('--sublinear_tf', type=bool, default=True)
    parser.add_argument('--kernel', type=str, default='sigmoid')
    parser.add_argument('--C', type=float, default=1)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--show_cm_matrix', type=bool, default=False)
    args = parser.parse_args()

    built_vocab('data/full_data/data.csv', cutoff=25)
    np.random.seed(args.seed)
    clf = Classifier.from_csv(args, args.train_file)
    clf.train()
    clf.evaluate(args.test_file, args.show_cm_matrix)

if __name__ == '__main__':
    main()
    print('Done!')


