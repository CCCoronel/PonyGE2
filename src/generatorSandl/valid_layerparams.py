# Dicionário para validação semântica de arquiteturas SANDL.
# Mapeia o nome de cada tipo de camada a uma lista de seus hiper-parâmetros
# válidos, conforme o uso comum em frameworks de deep learning.

VALID_LAYER_PARAMS = {
    # Camadas de Pooling
    "avg_pooling": [
        "pool_size",
        "strides",
        "padding"
    ],
    "global_pooling": [
        "pool_type"
    ],
    "max_pooling": [
        "pool_size",
        "strides",
        "padding"
    ],

    # Camadas de Normalização
    "batch_norm": [
        "axis",
        "momentum",
        "epsilon"
    ],
    "instance_norm": [
        "axis",
        "epsilon"
    ],
    "layer_norm": [
        "axis",
        "epsilon"
    ],

    # Camadas Convolucionais
    "conv1d": [
        "activation",
        "dilation_rate",
        "dropout",
        "filters",
        "kernel_init",
        "kernel_size",
        "l1",
        "l2",
        "padding",
        "strides"
    ],
    "tcn": [
        "activation",
        "dilations",
        "dropout",
        "filters",
        "kernel_init",
        "kernel_size",
        "l1",
        "l2"
    ],

    # Camadas Recorrentes e de Atenção
    "attention": [
        "dropout",
        "heads",
        "units"
    ],
    "esn": [
        "activation",
        "input_scaling",
        "leak_rate",
        "spectral_radius",
        "units"
    ],
    "gru": [
        "activation",
        "dropout",
        "kernel_init",
        "l1",
        "l2",
        "recurrent_dropout",
        "units"
    ],
    "lstm": [
        "activation",
        "dropout",
        "kernel_init",
        "l1",
        "l2",
        "recurrent_dropout",
        "units"
    ],

    # Camada Densa (Core)
    "dense": [
        "activation",
        "dropout",
        "kernel_init",
        "l1",
        "l2",
        "units"
    ]
}

# Exemplo de como usar o dicionário para validar um parâmetro
def is_param_valid_for_layer(layer_type: str, param: str) -> bool:
    """Verifica se um hiper-parâmetro é válido para um dado tipo de camada."""
    if layer_type in VALID_LAYER_PARAMS:
        return param in VALID_LAYER_PARAMS[layer_type]
    return False

# Testes de exemplo
print(f"É 'units' válido para 'dense'? {is_param_valid_for_layer('dense', 'units')}")
print(f"É 'strides' válido para 'lstm'? {is_param_valid_for_layer('lstm', 'strides')}")

