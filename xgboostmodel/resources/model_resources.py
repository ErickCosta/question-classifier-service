
import pandas as pd
import pickle
from nltk.corpus import stopwords
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


import warnings
warnings.filterwarnings('ignore')

def clean_text(df, text_field):
    df[text_field] = df[text_field].str.replace('\d', '')
    df[text_field] = df[text_field].str.replace('[\W]+\º', '')
    df[text_field] = df[text_field].str.replace('[\_]', '')
    df[text_field] = df[text_field].str.replace('[{}⅓\/()-+:.-@#$%¨&*=*βα–²Â?!“”ˆπ⅔]', '')
    df[text_field] = df
    return df

def train_model(label, params):
    questions = pd.read_csv("data/data").drop(['skills', 'type'], axis=1)

    clean_questions = clean_text(questions, 'text')

    list_labels = questions[['t_collect', 't_analysis', 't_representation',
                             't_decomposition', 't_algorithms', 't_abstraction',
                             't_automation', 't_parallelization', 't_simulation']]

    vectorizer = TfidfVectorizer(stop_words=stopwords.words("portuguese"), analyzer='word', ngram_range=(1, 2))

    data_train = vectorizer.fit_transform(clean_questions.text.str.lower())
    data_target = list_labels[label]

    model = XGBClassifier(params=params)

    model.fit(data_train, data_target)

    pickle.dump(model, open('trained_models/model_'+label, 'wb'))

def test_data(data, label):
    questions = pd.read_csv("data/data").drop(['skills', 'type'], axis=1)

    clean_questions = clean_text(questions, 'text')

    vectorizer = TfidfVectorizer(stop_words=stopwords.words("portuguese"), analyzer='word', ngram_range=(1, 2))

    vectorizer.fit_transform(clean_questions.text.str.lower())

    model = pickle.load(open('trained_models/model_'+label, "rb"))

    new_data = pd.DataFrame([data], columns=['text'])

    new_data = clean_text(new_data, 'text').text.str.lower()

    data_test = vectorizer.transform(new_data)

    response = model.predict(data_test)

    return response[0]


