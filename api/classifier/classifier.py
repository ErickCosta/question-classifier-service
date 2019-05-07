import pandas as pd
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import os


import warnings
warnings.filterwarnings('ignore')

def clean_text(df, text_field):
    df[text_field] = df[text_field].str.replace('\d', '')
    df[text_field] = df[text_field].str.replace('[\W]+\º', '')
    df[text_field] = df[text_field].str.replace('[\_]', '')
    df[text_field] = df[text_field].str.replace('[{}⅓\/()-+:.-@#$%¨&*=*βα–²Â?!“”ˆπ⅔]', '')
    df[text_field] = df
    return df

def test_data(data, label):
    print(os.path.abspath("data"))

    questions = pd.read_csv(os.path.abspath("classifier\data_base\data")).drop(['skills', 'type'], axis=1)

    clean_questions = clean_text(questions, 'text')

    vectorizer = TfidfVectorizer(stop_words=stopwords.words("portuguese"), analyzer='word', ngram_range=(1, 2))

    vectorizer.fit_transform(clean_questions.text.str.lower())

    model = pickle.load(open(os.path.abspath('classifier/trained_models/model_'+label), "rb"))

    new_data = pd.DataFrame([data], columns=['text'])

    new_data = clean_text(new_data, 'text').text.str.lower()

    data_test = vectorizer.transform(new_data)

    response = model.predict(data_test)

    return response[0]

def getClassifier(question):

    response_collect = test_data(question, 't_collect')

    response_analysis = test_data(question, 't_analysis')

    response_representation = test_data(question, 't_representation')

    response_decomposition = test_data(question, 't_decomposition')

    response_algorithms = test_data(question, 't_algorithms')

    response_abstraction = test_data(question, 't_abstraction')

    response_automation = test_data(question, 't_automation')

    response_parallelization = test_data(question, 't_parallelization')

    response_simulation = test_data(question, 't_simulation')

    return [response_collect, response_analysis, response_representation,
            response_decomposition, response_algorithms, response_abstraction,
            response_automation, response_parallelization, response_simulation]