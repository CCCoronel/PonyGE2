# Arquivo: my_fitness_evaluator.py

import numpy as np
import tensorflow as tf
from lark import Lark
from utilities.fitness.get_data import get_data
from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff# Supondo a herança do PonyGE2
from generatorSandl.sandl_interpreter import SemanticValidator, SANDLInterpreter
from generatorSandl.model_builder import build_model_from_dict

# A classe que você irá registrar no PonyGE2
class SANDLFitness(base_ff):

    def __init__(self):
        """
        Inicializa a função de fitness, carregando a gramática e os dados
        apenas uma vez.
        """
        # Inicializa a classe base do PonyGE2
        super().__init__()

        # --- Etapa de Setup Único ---
        print("INFO: Carregando a gramática Lark...")
        with open("sandl.lark", 'r', encoding='utf-8') as f:
            self.lark_parser = Lark(f.read(), parser='lalr', start='start')

        print("INFO: Carregando dados de treinamento e teste...")
        # Usando sua abordagem correta para carregar os dados
        self.X_train, self.y_train, self.X_test, self.y_test = \
            get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])
        
        # Você pode também guardar parâmetros importantes do Keras aqui
        # self.epochs = params.get('FITNESS_EPOCHS', 3)
        # self.batch_size = params.get('BATCH_SIZE', 256)
        
    def evaluate(self, phenotype: str):
        """
        Este é o método que o PonyGE2 chama para cada indivíduo (fenótipo).
        Ele executa a validação, construção, treinamento e avaliação.
        """
        try:
            # --- VALIDAÇÃO (Sintática e Semântica) ---
            parsed_tree = self.lark_parser.parse(phenotype)
            validator = SemanticValidator()
            validator.visit(parsed_tree)
            if validator.errors:
                print(f"AVISO: Indivíduo inválido (semântica). Penalizando.")
                return -1 # Pior fitness

            # --- INTERPRETAÇÃO E CONSTRUÇÃO ---
            interpreter = SANDLInterpreter()
            model_dict = interpreter.transform(parsed_tree)
            keras_model = build_model_from_dict(model_dict)

            # --- TREINAMENTO E AVALIAÇÃO ---
            keras_model.compile(
                optimizer='adam',
                loss='mean_squared_error', # Adapte para seu problema
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

            history = keras_model.fit(
                self.X_train, self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(self.X_test, self.y_test),
                verbose=0 # Essencial para não poluir o log
            )
            
            # O fitness pode ser o erro de validação (para minimizar)
            # ou a acurácia (para maximizar)
            val_rmse_history = history.history['val_root_mean_squared_error']
            fitness_value = min(val_rmse_history) # Queremos o menor erro obtido
            
            print(f"INFO: Indivíduo avaliado. Fitness (menor val_rmse) = {fitness_value:.6f}")
            
            # MUDANÇA 3: Retornar o valor do erro para minimização
            return fitness_value
            
            # PonyGE2 geralmente minimiza o fitness por padrão.
            # Se você estiver usando acurácia, precisaria usar "MAXIMIZE_FITNESS": True

        except Exception as e:
            print(f"AVISO: Indivíduo inválido (exceção durante o processo: {e}). Penalizando.")
            # Retorna um valor de fitness muito ruim
            return float('inf') # Infinito é um bom valor de penalidade para minimização