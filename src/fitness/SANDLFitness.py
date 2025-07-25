# Arquivo: my_fitness_evaluator.py
import os
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
        Inicializa a função de fitness.
        A importação de 'params' é feita aqui para evitar ciclos.
        """
        super().__init__()
        
        # --- IMPORTAÇÃO TARDIA (LAZY IMPORT) ---
        # Neste ponto, o PonyGE2 já carregou 'parameters.py' completamente.
        # Esta importação agora é segura e não circular.
        from algorithm.parameters import params
        
        # --- O resto do seu __init__ continua como antes ---
        current_dir = os.path.dirname(__file__)
        validation_grammar_path = os.path.join(current_dir, '..', 'generatorSandl', 'sandl.lark')

        # Normaliza o caminho para remover os '..' (boa prática)
        validation_grammar_path = os.path.normpath(validation_grammar_path)
        with open(validation_grammar_path, 'r', encoding='utf-8') as f:
            self.lark_parser = Lark(f.read(), parser='lalr', start='start')

        print("INFO (Fitness): Carregando dados de treinamento e teste...")
        try:
            self.X_train, self.y_train, self.X_test, self.y_test = \
                get_data(params['DATASET_TRAIN'], params['DATASET_TEST'])   
        except Exception as e:
            print(f"ERRO ao carregar os dados: {e}")
            raise RuntimeError("Falha ao carregar os dados de treinamento e teste. Verifique os caminhos e formatos dos arquivos.")

    def evaluate(self, phenotype: str, **kwargs):
        """
        Avalia um único fenótipo SANDL.
        """
        try:
            print(f"INFO (Fitness): Avaliando fenótipo: {phenotype[:50]}...")  # Exibe os primeiros 50 caracteres do fenótipo
            # 1. Validação
            parsed_tree = self.lark_parser.parse(phenotype)
            validator = SemanticValidator()
            validator.visit(parsed_tree)
            if validator.errors:
                return float('inf')

            print("INFO (Fitness): Fenótipo validado com sucesso.")
            # 2. Interpretação e Construção
            interpreter = SANDLInterpreter() 
            model_dict = interpreter.transform(parsed_tree)
            keras_model = build_model_from_dict(model_dict)

            print("INFO (Fitness): Modelo Keras construído a partir do fenótipo.")
            # 3. Compilação e Treinamento
            keras_model.compile(
                optimizer='adam',
                loss='mean_squared_error',
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )
            history = keras_model.fit(
                self.X_train, self.y_train, epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(self.X_test, self.y_test), verbose=0
            )
            
            # 4. Retorno do Fitness
            fitness_value = min(history.history['val_root_mean_squared_error'])
            return fitness_value

        except Exception:
            return float('inf')