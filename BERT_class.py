from sklearn.feature_extraction.text import TfidfVectorizer


class BERTFineTune:
    def __init__(self):
        return None

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        return self.clf

    def get_accuracy(self, X_test, y_test):
        print("model score: %.3f" % self.clf.score(X_test, y_test))
    
    def get_preds(self, test_set):
        preds = [x[0] for x in self.clf.predict(test_set)]
        return preds


    
# if __name__ == '__main__':
#     weights = [0.59, 0.74, 0.77, 0.89]
#     cat = CatBoost('text', 'years_required', weights)