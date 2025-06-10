from fitness.base_ff_classes.base_ff import base_ff
from utilities.fitness.get_data import get_data
from algorithm.parameters import params
from utilities.fitness.error_metric import eval
from fitness.supervised_learning.supervised_learning import supervised_learning

class time_series(supervised_learning):
    """
    Basic fitness function template for writing new fitness functions. This
    basic template inherits from the base fitness function class, which
    contains various checks and balances.
    
    Note that all fitness functions must be implemented as a class.
    
    Note that the class name must be the same as the file name.
    
    Important points to note about base fitness function class from which
    this template inherits:
    
      - Default Fitness values (can be referenced as "self.default_fitness")
        are set to NaN in the base class. While this can be over-written,
        PonyGE2 works best when it can filter solutions by NaN values.
    
      - The standard fitness objective of the base fitness function class is
        to minimise fitness. If the objective is to maximise fitness,
        this can be over-written by setting the flag "maximise = True".
    
    """

    # The base fitness function class is set up to minimise fitness.
    # However, if you wish to maximise fitness values, you only need to
    # change the "maximise" attribute here to True rather than False.
    # Note that if fitness is being minimised, it is not necessary to
    # re-define/overwrite the maximise attribute here, as it already exists
    # in the base fitness function class.
    maximise = False
    fitness_cache = {}

    def __init__(self, train, test, lookback=12, max_params=10000, early_stopping_patience=10):
        """
        All fitness functions which inherit from the bass fitness function
        class must initialise the base class during their own initialisation.
        """
        # Initialise base fitness function class.
        super().__init__()
        
        train, test = get_data(train, test)
        # Supondo que você já tenha train e test:
        train, test = self.normalize_data(train, test, method='standard')
        self.X_train, self.y_train = train
        self.X_test, self.y_test = test
        self.lookback = lookback
        self.max_params = max_params
        self.early_stopping_patience = early_stopping_patience


    def evaluate(self, ind, **kwargs):
        """
        Default fitness execution call for all fitness functions. When
        implementing a new fitness function, this is where code should be added
        to evaluate target phenotypes.
        
        There is no need to implement a __call__() method for new fitness
        functions which inherit from the base class; the "evaluate()" function
        provided here allows for this. Implementing a __call__() method for new
        fitness functions will over-write the __call__() method in the base
        class, removing much of the functionality and use of the base class.
                
        :param ind: An individual to be evaluated.
        :param kwargs: Optional extra arguments.
        :return: The fitness of the evaluated individual.
        """
        """
        Avalia a aptidão do indivíduo, usando cache e reparos se necessário.
        """
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, LSTM, GRU
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np

        try:
            # Passo 1: Obter uma chave única para o indivíduo
            chave_individuo = str(ind.arvore)

            # Passo 2: Verificar se já está no cache
            if chave_individuo in time_series.fitness_cache:
                return time_series.fitness_cache[chave_individuo]

            # Passo 3: Reparar a arquitetura, se necessário
            arvore_reparada = self._reparar_arvore(ind.arvore)
            ind.arvore = arvore_reparada  # Atualiza com versão corrigida

            # Passo 4: Construir modelo a partir da árvore reparada
            modelo = self._construir_modelo_a_partir_da_arvore(arvore_reparada)

            # Passo 5: Compilar e treinar com early stopping
            modelo.compile(optimizer='adam', loss='mse')
            historico = modelo.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_test, self.y_test),
                epochs=100,
                batch_size=32,
                callbacks=[EarlyStopping(patience=self.early_stopping_patience)],
                verbose=0
            )

            # Passo 6: Prever e calcular métricas
            y_pred = modelo.predict(self.X_test, verbose=0).flatten()
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            r2 = r2_score(self.y_test, y_pred)

            # Passo 7: Penalizações
            num_parametros = modelo.count_params()

            penalidade_complexidade = 0.0001 * max(0, num_parametros - self.max_params)

            erro_treino = min(historico.history['loss'])
            erro_validacao = min(historico.history['val_loss'])
            diferenca = erro_validacao - erro_treino
            penalidade_overfit = 0.5 * max(0, diferenca)

            # Composição final da aptidão
            aptidao_bruta = 1 / (1 + rmse)
            aptidao_final = aptidao_bruta / (1 + penalidade_complexidade + penalidade_overfit)

            # Salvar no cache
            time_series.fitness_cache[chave_individuo] = aptidao_final

            # Armazenar métricas extras no indivíduo (opcional)
            ind.rmse = rmse
            ind.r2 = r2
            ind.parametros = num_parametros
            ind.overfit_score = diferenca

            return aptidao_final

        except Exception as e:
            print(f"Erro ao avaliar indivíduo {ind}: {e}")
            return 0.001  # Penalidade leve para erros graves
        
        def _reparar_arvore(self, arvore):
          """
          Realiza correções automáticas na árvore para garantir validade mínima.
          
          Exemplos:
          - Garantir que a primeira camada tenha entrada compatível com janela temporal
          - Limitar número máximo de camadas
          - Ajustar número de neurônios se for excessivo
          """
        # Limite máximo de camadas
        MAX_CAMADAS = 5
        if len(arvore.camadas) > MAX_CAMADAS:
            arvore.camadas = arvore.camadas[:MAX_CAMADAS]

        # Garantir que a primeira camada tenha input_shape definido
        if hasattr(arvore.camadas[0], 'input_shape'):
            arvore.camadas[0].input_shape = (self.lookback, 1)

        # Limitar número de neurônios por camada
        MAX_NEURONIOS = 128
        for no in arvore.camadas:
            if no.tipo in ['Dense', 'LSTM', 'GRU'] and isinstance(no.args[0], int):
                no.args = (min(no.args[0], MAX_NEURONIOS),) + no.args[1:]

        return arvore

        def _construir_modelo_a_partir_da_arvore(self, arvore):
            """
            Converte uma árvore de derivação em uma rede neural Keras.
            """
            modelo = Sequential()

            for no in arvore.camadas:
                tipo_camada = no.tipo
                args = no.args

                if tipo_camada == 'Dense':
                    modelo.add(Dense(*args))
                elif tipo_camada == 'LSTM':
                    modelo.add(LSTM(*args, return_sequences=False if len(modelo.layers) == 0 else True))
                elif tipo_camada == 'GRU':
                    modelo.add(GRU(*args, return_sequences=False if len(modelo.layers) == 0 else True))
                else:
                    raise ValueError(f"Camada desconhecida: {tipo_camada}")

            return modelo

        # Evaluate the fitness of the phenotype
        fitness = eval(ind.phenotype)

        return fitness
    
    def normalize_data(train_data, test_data, method='standard'):
      """
      Normaliza os dados de treino e teste com base nos dados de treino.
      
      Parâmetros:
          train_data (array-like): Dados de treino crus.
          test_data (array-like): Dados de teste crus.
          method (str): Tipo de normalização: 'standard' (z-score) ou 'minmax'.

      Retorna:
          tuple: (train_normalized, test_normalized)
      """
      if method == 'standard':
          mean = train_data.mean(axis=0)
          std = train_data.std(axis=0)
          train_norm = (train_data - mean) / std
          test_norm = (test_data - mean) / std

      elif method == 'minmax':
          min_val = train_data.min(axis=0)
          max_val = train_data.max(axis=0)
          train_norm = (train_data - min_val) / (max_val - min_val)
          test_norm = (test_data - min_val) / (max_val - min_val)

      else:
          raise ValueError("Método de normalização inválido: use 'standard' ou 'minmax'.")

      return train_norm, test_norm
