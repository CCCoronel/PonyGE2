import argparse
from lark import Lark, Tree, Visitor, LarkError, Token

try:
    from valid_layerparams import VALID_LAYER_PARAMS
except ImportError:
    print("ERRO FATAL: O arquivo 'valid_layer_params.py' não foi encontrado. Certifique-se de que ele está no mesmo diretório.")
    exit(1)

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
        

    def layer_param(self, tree: Tree):
        if not self.current_layer_type:
            # Esta verificação é crucial. Se não estamos dentro de uma hidden_layer, não fazemos nada.
            return

        param_details_tree = tree.children[0]
        param_name_token = param_details_tree.children[0]
        param_value_node = param_details_tree.children[2]
        
        param_name = param_name_token.value
        line, col = param_name_token.line, param_name_token.column

        value_node = param_value_node
        while isinstance(value_node, Tree):
            value_node = value_node.children[0]
        value = value_node.value

        allowed_params = VALID_LAYER_PARAMS.get(self.current_layer_type)
        if allowed_params is None:
             self.errors.append(f"Erro de Configuração (L{line}): '{self.current_layer_type}' não está em valid_layer_params.py")
             return

        if param_name not in allowed_params:
            self.errors.append(
                f"Erro Estrutural (L{line}:C{col}): Parâmetro '{param_name}' não é válido para camada '{self.current_layer_type}'."
            )
        else:
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

def run_validation(grammar_filepath: str, sandl_filepath: str):
    print(f"\n--- Iniciando validação de '{sandl_filepath}' ---")
    
    try:
        with open(grammar_filepath, 'r', encoding='utf-8') as f:
            lark_parser = Lark(f.read(), parser='lalr', start='start')
        
        with open(sandl_filepath, 'r', encoding='utf-8') as f:
            sandl_content = f.read()

        print("1. Verificando a sintaxe...")
        parsed_tree = lark_parser.parse(sandl_content)
        print("  Sintaxe está correta.")

        print("2. Verificando a semântica (estrutura e valores)...")
        validator = SemanticValidator()
        validator.visit(parsed_tree)

        if not validator.errors:
            print(" Semântica está correta.")
            print("\n" + "="*30)
            print("ARQUIVO SANDL VÁLIDO")
            print("="*30)
        else:
            print("  Encontrados erros semânticos:")
            for error in sorted(validator.errors):
                print(f"     - {error}")
            print("\n" + "="*30)
            print("ARQUIVO SANDL INVÁLIDO")
            print("="*30)

    except FileNotFoundError as e:
        print(f"\nERRO FATAL: Arquivo não encontrado: {e.filename}")
    except LarkError as e:
        print(f"\nERRO DE SINTAXE:\n   Detalhes: {e}\n" + "="*30 + "\n ARQUIVO SANDL INVÁLIDO \n" + "="*30)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nERRO INESPERADO: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valida um arquivo .sandl.")
    parser.add_argument("grammar_file", type=str, help="Caminho para o arquivo .lark da gramática.")
    parser.add_argument("sandl_file", type=str, help="Caminho para o arquivo .sandl a ser validado.")
    args = parser.parse_args()
    
    run_validation(args.grammar_file, args.sandl_file)