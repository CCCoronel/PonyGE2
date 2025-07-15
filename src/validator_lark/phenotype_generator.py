# Arquivo: phenotype_generator.py (versão refatorada)

import argparse
import random
import re
from lark import Lark
from typing import List, Set

# Importa as regras de geração do arquivo de configuração
from generation_rules import GENERATION_RULES

class PhenotypeGenerator:
    """
    Gera arquiteturas SANDL (fenótipos) a partir de uma gramática Lark.
    Este gerador é "structure-aware", controlando a lógica de repetição
    diretamente em Python para evitar recursão infinita.
    """

    def __init__(self, grammar: Lark, max_hidden_layers: int, max_params_per_layer: int):
        self.grammar = grammar
        self.max_hidden_layers = max_hidden_layers
        self.max_params_per_layer = max_params_per_layer
        
        self._rules_map = {}
        for rule in self.grammar.rules:
            if rule.expansion and isinstance(rule.expansion[0], list):
                self._rules_map[rule.origin.name] = rule.expansion
            else:
                self._rules_map[rule.origin.name] = [rule.expansion]

        self._terminal_map = {t.name: t.pattern.value for t in self.grammar.terminals}

    def _get_random_choice(self, rule_name: str) -> str:
        """Expande uma regra simples de alternativas (ex: A | B | C) e retorna a string."""
        if rule_name not in self._rules_map:
            return self._terminal_map.get(rule_name, "")
        
        expansion = random.choice(self._rules_map[rule_name])
        parts = [self._get_random_choice(sub.name) for sub in expansion]
        return " ".join(parts)

    def _get_value(self, value_type: str, param_context: str) -> str:
        """Gera um valor para INT ou FLOAT, usando regras ou um fallback aleatório."""
        if value_type in ("INT", "FLOAT") and param_context in GENERATION_RULES:
            return str(GENERATION_RULES[param_context]())
        
        if value_type == "INT": return str(random.randint(8, 256))
        if value_type == "FLOAT": return f"{random.uniform(0.01, 0.5):.4f}"
        return ""

    def _generate_params(self, param_rule_name: str, used_params: Set[str]) -> List[str]:
        """Gera uma lista de strings de parâmetros para uma camada."""
        param_strings = []
        all_param_options = self._rules_map[param_rule_name]
        
        # Embaralha as opções para não gerar sempre os mesmos primeiros parâmetros
        random.shuffle(all_param_options)

        for _ in range(self.max_params_per_layer):
            for param_rule in all_param_options:
                param_name_terminal = param_rule[0]
                param_key = param_name_terminal.name

                if param_key not in used_params:
                    used_params.add(param_key)
                    param_name_str = self._terminal_map[param_key]
                    value_symbol = param_rule[2]

                    if value_symbol.name in ("INT", "FLOAT"):
                        value_str = self._get_value(value_symbol.name, param_key)
                    elif value_symbol.name == "dilation_list":
                        num_dilations = random.randint(1, 4)
                        ints = [self._get_value("INT", "DILATION_RATE") for _ in range(num_dilations)]
                        value_str = f"[{', '.join(ints)}]"
                    else: # É uma regra de escolha como activation_fn
                        value_str = self._get_random_choice(value_symbol.name)
                    
                    param_strings.append(f"{param_name_str} = {value_str};")
                    # Break para o loop interno para tentar o próximo param
                    break
            else:
                # Se o loop interno terminou sem break, não há mais params para adicionar
                break

        return param_strings

    def _generate_input_layer(self) -> str:
        params = []
        # Para a camada de entrada, os parâmetros são fixos e obrigatórios
        params.append(f"features = {self._get_value('INT', 'FEATURES')};")
        params.append(f"sequence_length = {self._get_value('INT', 'SEQUENCE_LENGTH')};")
        body = "\n        ".join(params)
        return f"input {{\n        {body}\n    }}"

    def _generate_output_layer(self) -> str:
        params = []
        # Para a camada de saída, unidades e ativação são tipicamente necessárias
        params.append(f"units = {self._get_value('INT', 'UNITS')};")
        params.append(f"activation = {self._get_random_choice('activation_fn')};")
        # Dropout é opcional
        if random.random() < 0.5:
             params.append(f"dropout = {self._get_value('FLOAT', 'DROPOUT')};")
        body = "\n        ".join(params)
        return f"output {{\n        {body}\n    }}"

    def _generate_hidden_layer(self) -> str:
        layer_type = self._get_random_choice('layer_type')
        used_params = set()
        
        # Alguns parâmetros são essenciais para certas camadas
        if layer_type in ["dense", "lstm", "gru", "esn"]:
            used_params.add("UNITS")
            param_str = f"units = {self._get_value('INT', 'UNITS')};"
        elif layer_type in ["conv1d", "tcn"]:
            used_params.add("FILTERS")
            param_str = f"filters = {self._get_value('INT', 'FILTERS')};"
        else: # Para outras camadas como pooling, norm, etc. não há um param obrigatório
            param_str = ""

        # Gera outros parâmetros aleatórios
        other_params = self._generate_params('layer_param', used_params)
        
        all_params = [p for p in ([param_str] + other_params) if p]
        body = "\n        ".join(all_params)
        return f"{layer_type} {{\n        {body}\n    }}"

    def _format_output(self, text: str) -> str:
        """Aplica uma indentação simples para legibilidade."""
        # A geração agora controla as quebras de linha, então a formatação é mais simples.
        # Esta função apenas garante que a indentação esteja correta.
        formatted_text = ""
        indent_level = 0
        for line in text.split('\n'):
            line = line.strip()
            if not line: continue
            if "}" in line and "{" not in line:
                indent_level = max(0, indent_level - 1)
            
            formatted_text += "    " * indent_level + line + "\n"
            
            if "{" in line and "}" not in line:
                indent_level += 1
        return formatted_text

    def generate(self) -> str:
        """Gera uma arquitetura SANDL completa e formatada."""
        input_layer_str = self._generate_input_layer()
        output_layer_str = self._generate_output_layer()
        
        num_hidden = random.randint(1, self.max_hidden_layers)
        hidden_layers = [self._generate_hidden_layer() for _ in range(num_hidden)]
        hidden_layers_str = "\n\n    ".join(hidden_layers)
        
        raw_output = f"""neuralnet {{
    {input_layer_str}

    {hidden_layers_str}

    {output_layer_str}
}}"""
        return self._format_output(raw_output)

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera um fenótipo SANDL aleatório a partir de uma gramática Lark."
    )
    parser.add_argument(
        "grammar_file", type=str, help="Caminho para o arquivo .lark da gramática."
    )
    parser.add_argument(
        "--max-layers", type=int, default=6, help="Número máximo de camadas ocultas."
    )
    parser.add_argument(
        "--max-params", type=int, default=5, help="Número máximo de parâmetros por camada."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Semente para o gerador aleatório."
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        # print(f"INFO: Semente do gerador aleatório definida como: {args.seed}")

    try:
        with open(args.grammar_file, 'r', encoding='utf-8') as f:
            lark_grammar = Lark(f.read(), parser='lalr', start='start')

        generator = PhenotypeGenerator(lark_grammar, 
                                       max_hidden_layers=args.max_layers, 
                                       max_params_per_layer=args.max_params)

        # print("\n--- Exemplo de Fenótipo Gerado (Arquitetura SANDL) ---")
        random_architecture = generator.generate()
        print(random_architecture)
        # print("\n" + "-"*55)

    except FileNotFoundError:
        print(f"ERRO: O arquivo da gramática '{args.grammar_file}' não foi encontrado.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERRO: Ocorreu um erro inesperado: {e}")