from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
import numpy as np
import pickle


def model_fit_predict(clf, train_data, train_label, test_data, test_label):
    clf.fit(train_data, train_label)
    print("training accuracy: ", clf.score(train_data, train_label))
    print("testing accuracy: ", clf.score(test_data, test_label))
    test_predict = clf.predict(test_data)
    test_predict_prob = clf.predict_proba(test_data)
    print("testing auc score: ", roc_auc_score(test_label, test_predict_prob[:, 1]))
    print(classification_report(test_label, test_predict))
    return clf


def data_vect_fit(df, column, **kwargs):
    transform_collection = {}
    tf_idf_model = TfidfVectorizer(min_df=0.03, max_df=0.97, ngram_range=(1, 2))
    tf_idf_output = tf_idf_model.fit_transform(df[column])
    transform_collection['tf_idf'] = tf_idf_model
    result = tf_idf_output.toarray()
    print(result.shape)
    if kwargs.get('sentiment', None):
        sentiment_output = df[['polarity', 'subjectivity']].values
        result = np.concatenate([result, sentiment_output], axis=1)
        transform_collection['sentiment'] = 'blob'
        print(result.shape)
    if kwargs.get('topic_vect', None):
        vectorizer = CountVectorizer()
        count_vec = vectorizer.fit_transform(df[column])
        topic_num = kwargs.get('topic_num', 20)
        lda_model = LatentDirichletAllocation(n_components=topic_num, learning_method='online', n_jobs=-1)
        lda_output = lda_model.fit_transform(count_vec)
        transform_collection['topic_model'] = (vectorizer, lda_model)
        result = np.concatenate([result, lda_output], axis=1)
        print(result.shape)
    return result, transform_collection


def data_vect_transform(df, column, transform_collection: dict):
    result = []
    for key, value in transform_collection.items():
        if key == 'tf_idf':
            result.append(value.transform(df[column]).toarray())
        elif key == 'sentiment':
            df['polarity'] = df[column].apply(lambda x: TextBlob(x).sentiment.polarity)
            df['subjectivity'] = df[column].apply(lambda x: TextBlob(x).sentiment.subjectivity)
            result.append(df[['polarity', 'subjectivity']].values)
        else:
            vectorizer, lda_model = value
            count_vec = vectorizer.transform(df[column])
            lda_output = lda_model.transform(count_vec)
            result.append(lda_output)
    result = np.concatenate(result, axis=1)
    return result


def save_transform(transform_collection, filename):
    with open(filename, 'wb') as f:
        pickle.dump(transform_collection, f)


def load_transform(filename):
    with open(filename, 'rb') as f:
        transform = pickle.load(f)
    return transform
