from xgboostmodel.resources import model_resources


def training_models():
       model_resources.train_model('t_collect', {'gamma': 0.1, 'reg_alpha': 1e-05})

       model_resources.train_model('t_analysis', {'gamma': 1, 'reg_alpha': 1})

       model_resources.train_model('t_representation', {'gamma': 1, 'reg_alpha': 1})

       model_resources.train_model('t_decomposition', {'gamma': 0.1, 'reg_alpha': 0.1})

       model_resources.train_model('t_algorithms', {'gamma': 1e-05, 'reg_alpha': 1})

       model_resources.train_model('t_abstraction', {'gamma': 0.1, 'reg_alpha': 1})

       model_resources.train_model('t_automation', {'gamma': 1e-05, 'reg_alpha': 1e-05})

       model_resources.train_model('t_parallelization', {'gamma': 1e-05, 'reg_alpha': 1e-05})

       model_resources.train_model('t_simulation', {'gamma': 1e-05, 'reg_alpha': 1e-05})

def predict_question(question):

       response_collect = model_resources.test_data(question, 't_collect')

       response_analysis = model_resources.test_data(question, 't_analysis')

       response_representation = model_resources.test_data(question, 't_representation')

       response_decomposition = model_resources.test_data(question, 't_decomposition')

       response_algorithms = model_resources.test_data(question, 't_algorithms')

       response_abstraction = model_resources.test_data(question, 't_abstraction')

       response_automation = model_resources.test_data(question, 't_automation')

       response_parallelization = model_resources.test_data(question, 't_parallelization')

       response_simulation = model_resources.test_data(question, 't_simulation')

       return [response_collect, response_analysis, response_representation,
       response_decomposition, response_algorithms, response_abstraction,
       response_automation, response_parallelization, response_simulation]

question =  "Um forro retangular de tecido traz em sua etiqueta a informação de que encolherá após a primeira lavagem mantendo entretanto seu formato. A figura a seguir mostra as medidas originais do forro e o tamanho do encolhimento (x) no comprimento e (y) na largura. A expressão algébrica que representa a área do forro após ser lavado é (5 – x) (3 – y). Nestas condições a área perdida do forro após a primeira lavagem será expressa por"

print(predict_question(question))


