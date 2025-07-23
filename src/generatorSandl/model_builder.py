# Arquivo: model_builder.py
# Constrói modelos Keras/TensorFlow a partir da representação de dicionário do SANDL.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Mapeamento completo dos nomes de camadas do SANDL para as classes de camadas do Keras.
# Isso garante que todas as arquiteturas geradas pela gramática possam ser construídas.
LAYER_MAPPING = {
    # Camadas Padrão
    "dense": layers.Dense,
    
    # Camadas Recorrentes
    "lstm": layers.LSTM,
    "gru": layers.GRU,
    
    # Camadas Convolucionais
    "conv1d": layers.Conv1D,
    
    # Camadas de Atenção
    "attention": layers.Attention, # Nota: Attention pode requerer uso mais complexo
    
    # Camadas de Pooling
    "max_pooling": layers.MaxPooling1D,
    "avg_pooling": layers.AveragePooling1D,
    "global_pooling": layers.GlobalAveragePooling1D, # Uma escolha comum para "global_pooling"
    
    # Camadas de Normalização
    "batch_norm": layers.BatchNormalization,
    "layer_norm": layers.LayerNormalization,
    "instance_norm": None,  # Não existe nativamente no Keras. Requer tf-addons.
    
    # Camadas mais complexas ou customizadas (Exemplos)
    "tcn": None, # Temporal Convolutional Network - Geralmente é uma sequência de outras camadas.
    "esn": None, # Echo State Network - Requer uma implementação customizada.
}

def build_model_from_dict(model_dict: dict) -> keras.Model:
    """
    Constrói um modelo sequencial do Keras a partir de um dicionário
    gerado pelo interpretador SANDL.
    """
    model = keras.Sequential()
    is_first_layer = True

    # 1. Determinar o formato de entrada (input_shape)
    input_params = model_dict.get('input_layer', {})
    input_shape = None
    if 'features' in input_params:
        # O formato para camadas recorrentes/convolucionais é (timesteps, features)
        sequence_length = input_params.get('sequence_length') # Pode ser None
        features = input_params['features']
        input_shape = (sequence_length, features)

    # 2. Adicionar as camadas ocultas (Hidden Layers)
    hidden_layers = model_dict.get('hidden_layers', [])
    for layer_info in hidden_layers:
        # Faz uma cópia para não modificar o dicionário original
        params = layer_info.copy()
        layer_type_name = params.pop('type')
        
        # Obtém a classe da camada Keras a partir do mapeamento
        LayerClass = LAYER_MAPPING.get(layer_type_name)

        if LayerClass:
            # Se for a primeira camada do modelo, adiciona o input_shape
            if is_first_layer and input_shape:
                params['input_shape'] = input_shape
                is_first_layer = False
            
            # Adiciona a camada ao modelo, desempacotando os parâmetros
            # Ex: {'units': 64, 'activation': 'relu'} -> layers.Dense(units=64, activation='relu')
            model.add(LayerClass(**params))
        
        elif layer_type_name == "instance_norm":
            print("AVISO: 'instance_norm' não é uma camada Keras nativa e foi ignorada. Considere usar a biblioteca 'tensorflow-addons'.")
            continue # Pula para a próxima camada
        
        else:
            # Lança um erro se a camada for desconhecida ou não implementada
            raise ValueError(f"Tipo de camada desconhecido ou não implementado no construtor: '{layer_type_name}'")

    # 3. Adicionar a camada de saída (Output Layer)
    output_params = model_dict.get('output_layer')
    if output_params:
        # A camada de saída é quase sempre uma camada Densa
        # Se o modelo ainda não tiver recebido o input_shape (sem camadas ocultas),
        # ele deve ser adicionado aqui.
        if is_first_layer and input_shape:
            output_params['input_shape'] = input_shape
            is_first_layer = False
        
        model.add(layers.Dense(**output_params))
    
    # Se, ao final, nenhuma camada foi adicionada (arquitetura inválida),
    # lança um erro para penalizar o indivíduo na evolução.
    if not model.layers:
        raise ValueError("A arquitetura gerada não resultou em nenhuma camada no modelo Keras.")
        
    return model