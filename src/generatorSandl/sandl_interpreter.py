# Arquivo: sandl_interpreter.py
# Versão Final: Valida (sintática e semanticamente) e depois Interpreta um arquivo .sandl

import argparse
import json
from lark import Lark, Tree, Visitor, Transformer, LarkError, Token

# Tenta importar as regras de validação de um arquivo separado.
# Este arquivo é crucial para a validação semântica.
try:
    from valid_layerparams import VALID_LAYER_PARAMS
except ImportError:
    print("ERRO FATAL: O arquivo 'valid_layer_params.py' não foi encontrado.")
    print("Certifique-se de que ele existe no mesmo diretório e contém um dicionário chamado VALID_LAYER_PARAMS.")
    exit(1)

# Definição das regras para valores numéricos, usadas pelo SemanticValidator.
SEMANTIC_RULES = {
    "FLOAT_RANGES": {
        "dropout": (0.0, 1.0), "recurrent_dropout": (0.0, 1.0), "l1": (0.0, float('inf')),
        "l2": (0.0, float('inf')), "leak_rate": (0.0, 1.0), "momentum": (0.0, 1.0),
        "spectral_radius": (0.0, float('inf')), "input_scaling": (0.0, float('inf')),
        "epsilon": (1e-7, float('inf')),
    },
    "INT_RANGES": {
        "units": (1, float('inf')), "filters": (1, float('inf')), "heads": (1, float('inf')),
        "features": (1, float('inf')), "sequence_length": (1, float('inf')),
        "kernel_size": (1, float('inf')), "strides": (1, float('inf')),
        "pool_size": (1, float('inf')), "dilation_rate": (1, float('inf')),
    }
}


class SemanticValidator(Visitor):
    def __init__(self):
        super().__init__()
        self.errors = []
        self.current_layer_type = None

    def hidden_layer(self, tree: Tree):
        layer_type_node = next(tree.find_data('layer_type'), None)
        if layer_type_node:
            self.current_layer_type = layer_type_node.children[0].value
        else:
            self.current_layer_type = None
        
        for child in tree.children:
            if isinstance(child, Tree):
                self.visit(child)

    def layer_param(self, tree: Tree):
        if not self.current_layer_type: return

        param_name_token = tree.children[0]
        param_value_node = tree.children[2]
        
        param_name = param_name_token.value
        line, col = param_name_token.line, param_name_token.column
        
        if not (isinstance(param_value_node, Token) and param_value_node.type in ('INT', 'FLOAT')):
             return
        
        value = param_value_node.value

        allowed_params = VALID_LAYER_PARAMS.get(self.current_layer_type)
        if allowed_params is None:
             self.errors.append(f"Erro de Configuração (L{line}): '{self.current_layer_type}' não definido em valid_layer_params.py")
             return

        if param_name not in allowed_params:
            self.errors.append(
                f"Erro Estrutural (L{line}:C{col}): Parâmetro '{param_name}' não é válido para camada '{self.current_layer_type}'."
            )
        else:
            self._validate_value(param_name, value, line, col)
    
    def input_param(self, tree: Tree):
        param_name_token = tree.children[0]
        param_value_node = tree.children[2]
        param_name = param_name_token.value
        line, col = param_name_token.line, param_name_token.column
        value = param_value_node.value if isinstance(param_value_node, Token) else param_value_node.children[0].value
        self._validate_value(param_name, value, line, col)

    def output_param(self, tree: Tree):
        param_name_token = tree.children[0]
        param_value_node = tree.children[2]
        
        if isinstance(param_value_node, Tree) and param_value_node.data == 'activation_fn':
            return

        param_name = param_name_token.value
        line, col = param_name_token.line, param_name_token.column
        value = param_value_node.value if isinstance(param_value_node, Token) else param_value_node.children[0].value
        self._validate_value(param_name, value, line, col)

    def _validate_value(self, name: str, value, line: int, col: int):
        if name in SEMANTIC_RULES["FLOAT_RANGES"]:
            min_val, max_val = SEMANTIC_RULES["FLOAT_RANGES"][name]
            value = float(value)
            if not (min_val <= value < max_val):
                self.errors.append(f"Erro de Valor (L{line}:C{col}): O valor de '{name}' ({value}) está fora da faixa [{min_val}, {max_val}).")
        elif name in SEMANTIC_RULES["INT_RANGES"]:
            min_val, _ = SEMANTIC_RULES["INT_RANGES"][name]
            value = int(value)
            if value < min_val:
                self.errors.append(f"Erro de Valor (L{line}:C{col}): O valor de '{name}' ({value}) não pode ser menor que {min_val}.")

# ==============================================================================
# CLASSE DE INTERPRETAÇÃO - VERSÃO FINAL COM TODAS AS TRANSFORMAÇÕES
# ==============================================================================
class SANDLInterpreter(Transformer):
    # Converte tokens em valores Python nativos
    def INT(self, i): return int(i)
    def FLOAT(self, f): return float(f)

    # Converte tokens de string em strings Python
    def LINEAR(self, _): return "linear"
    def RELU(self, _): return "relu"
    def SIGMOID(self, _): return "sigmoid"
    def TANH(self, _): return "tanh"
    def SOFTMAX(self, _): return "softmax"
    def SWISH(self, _): return "swish"
    def GELU(self, _): return "gelu"
    def LEAKY_RELU(self, _): return "leaky_relu"
    def ELU(self, _): return "elu"
    
    def DENSE(self, _): return "dense"
    def LSTM(self, _): return "lstm"
    def GRU(self, _): return "gru"
    def CONV1D(self, _): return "conv1d"
    def TCN(self, _): return "tcn"
    def ATTENTION(self, _): return "attention"
    
    def GLOROT_UNIFORM(self, _): return "glorot_uniform"
    def GLOROT_NORMAL(self, _): return "glorot_normal"
    def HE_UNIFORM(self, _): return "he_uniform"
    def HE_NORMAL(self, _): return "he_normal"
    def RANDOM_NORMAL(self, _): return "random_normal"

    def CAUSAL(self, _): return "causal"
    def SAME(self, _): return "same"
    def VALID(self, _): return "valid"

    def AVG(self, _): return "avg"
    def MAX(self, _): return "max"
    def MIN(self, _): return "min"
    
    # --- CORREÇÃO FINAL ADICIONADA AQUI ---
    def layer_type(self, items): return items[0]
    # --------------------------------------

    # Transforma regras "container" em seu conteúdo processado
    def activation_fn(self, items): return items[0]
    def kernel_initializer(self, items): return items[0]
    def dilation_list(self, items): return [item for item in items[1:-1] if isinstance(item, int)]

    # Transforma regras de parâmetro em tuplas (chave, valor)
    def layer_param(self, items): return (items[0].type.lower(), items[2])
    def input_param(self, items): return (items[0].type.lower(), items[2])
    def output_param(self, items): return (items[0].type.lower(), items[2])
    
    # Transforma regras de camada em dicionários
    def input_layer(self, items):
        params = [item for item in items if isinstance(item, tuple)]
        return dict(params)

    def output_layer(self, items):
        params = [item for item in items if isinstance(item, tuple)]
        return dict(params)

    def hidden_layer(self, items):
        layer_type = items[0]
        params_list = [item for item in items if isinstance(item, tuple)]
        params_dict = dict(params_list)
        params_dict['type'] = layer_type
        return params_dict

    # CÓDIGO CORRIGIDO para o método ann
    def ann(self, items):
        # items[0] é o token NEURALNET
        # items[1] é o dicionário da camada de entrada
        # items[-1] é o dicionário da camada de saída
        # O que estiver no meio são as camadas ocultas
        return {
            "input_layer": items[1],
            "hidden_layers": items[2:-1],  # Pega a fatia do meio
            "output_layer": items[-1]
        }
    def start(self, items): return items[0]

# Função principal (run_interpreter) permanece a mesma.
def run_interpreter(grammar_filepath: str, sandl_filepath: str):
    print(f"\n--- Iniciando processo para '{sandl_filepath}' ---")
    
    try:
        with open(grammar_filepath, 'r', encoding='utf-8') as f:
            lark_parser = Lark(f.read(), parser='lalr', start='start')
        
        with open(sandl_filepath, 'r', encoding='utf-8') as f:
            sandl_content = f.read()

        print("1. Verificando a sintaxe...")
        parsed_tree = lark_parser.parse(sandl_content)
        print("  Sintaxe está correta.")

        print("2. Verificando a semântica (regras e valores)...")
        validator = SemanticValidator()
        validator.visit(parsed_tree)

        if validator.errors:
            print("  Encontrados erros semânticos. A interpretação foi abortada.")
            for error in sorted(validator.errors):
                print(f"     - {error}")
            print("\n" + "="*30 + "\nARQUIVO SANDL INVÁLIDO\n" + "="*30)
            return
        
        print("  Semântica está correta.")

        print("3. Interpretando a arquitetura...")
        interpreter = SANDLInterpreter()
        model_dict = interpreter.transform(parsed_tree)
        print("Interpretação concluída.")
        
        print("\n" + "="*50)
        print("ARQUIVO VÁLIDO E INTERPRETADO COM SUCESSO")
        print("--- Representação do Modelo em Python ---")
        print(json.dumps(model_dict, indent=4))
        print("="*50)

    except FileNotFoundError as e:
        print(f"\nERRO FATAL: Arquivo não encontrado: {e.filename}")
    except LarkError as e:
        if hasattr(e, 'orig_exc') and isinstance(e.orig_exc, (AttributeError, TypeError)):
            print(f"\nERRO DE INTERPRETAÇÃO: {e.orig_exc}")
            print("   Verifique se todos os métodos de transformação estão implementados na classe SANDLInterpreter.")
        else:
            print(f"\nERRO DE SINTAXE: {e}")
        print("\n" + "="*30 + "\nARQUIVO SANDL INVÁLIDO\n" + "="*30)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nERRO INESPERADO: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida e interpreta um arquivo .sandl.")
    parser.add_argument("grammar_file", type=str, help="Caminho para o arquivo .lark da gramática.")
    parser.add_argument("sandl_file", type=str, help="Caminho para o arquivo .sandl a ser processado.")
    args = parser.parse_args()
    
    run_interpreter(args.grammar_file, args.sandl_file)