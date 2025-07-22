# Arquivo: generation_config.py

import random
from typing import Dict, Callable, Any

"""
Este arquivo define as regras para a geração de valores numéricos "bem-comportados".
Cada chave do dicionário corresponde a um terminal definido no arquivo .lark 
(ex: "UNITS", "DROPOUT"). O valor associado é uma função que retorna um
número aleatório dentro de uma faixa ou conjunto predefinido.
"""

GENERATION_RULES: Dict[str, Callable[[], Any]] = {
    # --- Parâmetros que usam INT ---
    "UNITS":           lambda: random.choice([16, 32, 64, 128, 256, 512]),
    "FILTERS":         lambda: random.choice([16, 32, 64, 128]),
    "FEATURES":        lambda: random.choice([10, 20, 50, 100]),
    "SEQUENCE_LENGTH": lambda: random.choice([30, 50, 100, 200]),
    "HEADS":           lambda: random.choice([1, 2, 4, 8]),
    "KERNEL_SIZE":     lambda: random.choice([3, 5, 7, 9]),
    "STRIDES":         lambda: random.choice([1, 2]),
    "POOL_SIZE":       lambda: random.choice([2, 3, 4]),
    "DILATION_RATE":   lambda: random.choice([1, 2, 4, 8]),
    "AXIS":            lambda: random.choice([-1, 1]),
    # Regra genérica para INTs que não têm um contexto específico (como em dilation_list)
    "INT":             lambda: random.choice([1, 2, 4, 8]),
    
    # --- Parâmetros que usam FLOAT ---
    "DROPOUT":           lambda: round(random.uniform(0.1, 0.5), 3),
    "RECURRENT_DROPOUT": lambda: round(random.uniform(0.1, 0.5), 3),
    "L1":                lambda: f"{random.choice([1e-2, 1e-3, 1e-4]):.0e}",
    "L2":                lambda: f"{random.choice([1e-2, 1e-3, 1e-4]):.0e}",
    "LEAK_RATE":         lambda: round(random.uniform(0.1, 0.3), 2),
    "SPECTRAL_RADIUS":   lambda: round(random.uniform(0.9, 0.99), 2),
    "INPUT_SCALING":     lambda: round(random.uniform(0.5, 1.5), 2),
    "MOMENTUM":          lambda: round(random.uniform(0.9, 0.99), 2),
    "EPSILON":           lambda: f"{random.choice([1e-3, 1e-5, 1e-7]):.0e}",
}