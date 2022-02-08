from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class CatBoost:
    def __init__(self, text_col_name, num_col_name, class_weights):
        self.preprocess = ColumnTransformer(
     [("vectorizer", TfidfVectorizer(max_features=400, stop_words='english', token_pattern = "[a-zA-Z]{2,}", ngram_range = (1, 2), lowercase=True, strip_accents='unicode'), text_col_name),
      ("encoder", OneHotEncoder(handle_unknown='ignore'), [num_col_name])])
        self.clf = Pipeline(steps=[("preprocessor", self.preprocess), ("classifier", CatBoostClassifier(loss_function='MultiClass', eval_metric= 'MultiClass', random_state=1, class_weights=class_weights, learning_rate=0.7, iterations=50, depth=10))])
        return None

    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
        return self.clf

    def get_accuracy(self, X_test, y_test):
        print("model score: %.3f" % self.clf.score(X_test, y_test))
    
    def get_preds(self, test_set):
        preds = [x[0] for x in self.clf.predict(test_set)]
        return preds
    
    def return_new_json(self, dataframe, column_name, predictions):
        dataframe.loc[dataframe[column_name].isnull(), column_name] = predictions
        # assert len(df.loc[df[column_name].isnull()]) != 0, 'There has been an error. Not all values have been replaced.'
        dataframe.to_json('updated_data.json')
        print('Congratulations. Your file has been updated and saved to your project directory as "updated_data.json"!')

    
# if __name__ == '__main__':
#     weights = [0.59, 0.74, 0.77, 0.89]
#     cat = CatBoost('text', 'years_required', weights)