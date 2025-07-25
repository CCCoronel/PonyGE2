import argparse
import random
import re
from typing import Dict, List, Set

class SANDLGenerator:
    """
    Gera arquiteturas de redes neurais aleatórias baseadas em uma gramática BNF.
    Esta versão inclui lógica interna para garantir a geração de parâmetros mais
    ricos e formatação correta sem modificar a gramática original.
    """

    def __init__(self, grammar_filepath: str):
        self.grammar = self._parse_grammar(grammar_filepath)
        self.indent_level = 0

    def _parse_grammar(self, filepath: str) -> Dict[str, List[List[str]]]:
        grammar = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            content = ""
            for line in f:
                if not line.strip().startswith('#'):
                    content += line.strip() + " "

        rule_pattern = re.compile(r'(<\S+?>)\s*::=\s*(.*?)(?=\s*<\S+?>\s*::=|$)')

        for match in rule_pattern.finditer(content):
            non_terminal = match.group(1).strip()
            productions = match.group(2).strip()
            choices = []
            for choice_str in productions.split('|'):
                tokens = [token for token in choice_str.strip().split()]
                if tokens:
                    choices.append(tokens)
            if non_terminal and choices:
                grammar[non_terminal] = choices
        return grammar

    def _get_indent(self) -> str:
        """Retorna a string de indentação para o nível atual."""
        return "    " * self.indent_level

    def _generate_managed_params(self, symbol: str, depth: int, max_depth: int) -> str:
        """
        Gera uma lista rica e bem formatada de parâmetros para um tipo de camada,
        ignorando a recursão 'vazia' da gramática.
        """
        params_to_generate = []
        
        if symbol == '<input_params>':
            params_to_generate = self.grammar.get('<input_param>', [])
        
        elif symbol == '<output_params>':
            possible = self.grammar.get('<output_param>', [])
            # Garante 'units' e 'activation'
            params_to_generate = [p for p in possible if '<units_param>' in p or '<activation_param>' in p]
            # Adiciona talvez um dropout opcional
            if random.random() > 0.5:
                dropout_param = next((p for p in possible if '<dropout_param>' in p), None)
                if dropout_param:
                    params_to_generate.append(dropout_param)

        elif symbol == '<layer_params>':
            possible_params = self.grammar.get('<layer_param>', [])
            if not possible_params: return ""
            num_to_generate = random.randint(2, min(5, len(possible_params)))
            params_to_generate = random.sample(possible_params, num_to_generate)

        if not params_to_generate:
            return ""

        # Expande cada parâmetro escolhido
        expanded_params = [self._expand(p[0], depth, max_depth) for p in params_to_generate]

        # Formata a lista em uma string final com ponto e vírgula e quebras de linha
        output_parts = []
        for i, p_str in enumerate(expanded_params):
            output_parts.append(p_str + ";")
            # Se não for o último, adiciona quebra de linha e indentação para o próximo
            if i < len(expanded_params) - 1:
                output_parts.append("\n" + self._get_indent())
        
        return "".join(output_parts)

    def _expand(self, symbol: str, depth: int, max_depth: int) -> str:
        """Expande recursivamente um símbolo da gramática, com formatação e lógica de geração especial."""
        # Delega a geração de listas de parâmetros para a lógica gerenciada
        if symbol in ['<input_params>', '<output_params>', '<layer_params>']:
            return self._generate_managed_params(symbol, depth, max_depth)
            
        if symbol not in self.grammar:
            return symbol.strip('"')

        # Lógica de controle de profundidade
        is_recursive = any(symbol in p for p in self.grammar[symbol])
        if is_recursive and depth >= max_depth:
            non_recursive_productions = [p for p in self.grammar[symbol] if symbol not in p]
            if not non_recursive_productions: return ""
            production = random.choice(non_recursive_productions)
        else:
            production = random.choice(self.grammar[symbol])

        if production == ['<<empty>>']: return ""

        parts = []
        for part_symbol in production:
            if part_symbol == '"{"':
                self.indent_level += 1
                if parts and parts[-1] == " ": parts.pop()
                parts.append(" {\n" + self._get_indent())
                continue

            elif part_symbol == '"}"':
                self.indent_level -= 1
                parts.append("\n" + self._get_indent() + "}")
                continue

            # O ponto e vírgula agora é tratado pela _generate_managed_params
            if part_symbol == '";"': continue

            expanded_part = self._expand(part_symbol, depth + 1 if part_symbol == symbol else depth, max_depth)

            if expanded_part:
                # --- LÓGICA DE FORMATAÇÃO CORRIGIDA ---
                prefix = ""
                # Se o último item adicionado foi um bloco '}', o próximo deve
                # começar em uma nova linha com a indentação correta.
                if parts and parts[-1].strip().endswith('}'):
                    prefix = "\n" + self._get_indent()
                
                parts.append(prefix + expanded_part)

        return "".join(parts)

    def generate(self, max_depth: int = 15) -> str:
        """Gera e formata uma definição de arquitetura SANDL completa."""
        if not self.grammar:
            return "Erro: A gramática não foi carregada corretamente."

        self.indent_level = 0
        raw_output = self._expand('<ann>', 0, max_depth)
        
        # O pós-processamento agora é mais simples
        lines = [line.rstrip() for line in raw_output.split('\n')]
        clean_output = "\n".join(line for line in lines if line.strip())
        
        return clean_output

def generate_single_architecture(grammar_path: str, seed: int = None, max_depth: int = 15) -> str:
    """
    Instancia o gerador e produz uma única arquitetura SANDL.
    """
    if seed is not None:
        random.seed(seed)
    
    try:
        generator = SANDLGenerator(grammar_path)
        architecture = generator.generate(max_depth=max_depth)
        return architecture
    except Exception as e:
        print(f"ERRO durante a geração da arquitetura: {e}")
        return ""
    
# --- Bloco de Execução Principal (sem alterações) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera uma arquitetura SANDL aleatória e a salva em um arquivo."
    )
    parser.add_argument(
        "grammar_file",
        type=str,
        help="O caminho para o arquivo .bnf contendo a gramática SANDL."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="generated_architecture.sandl",
        help="Nome do arquivo para salvar a arquitetura gerada. Padrão: generated_architecture.sandl"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A raiz para o gerador de números aleatórios."
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=15,
        help="Profundidade máxima de recursão."
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        print(f"INFO: Raiz do gerador aleatório definida como: {args.seed}")

    try:
        generator = SANDLGenerator(args.grammar_file)
        random_architecture = generator.generate(max_depth=args.max_depth)
        
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(random_architecture)
            print(f"INFO: Arquitetura salva com sucesso em '{args.output_file}'")
        except IOError as e:
            print(f"ERRO: Não foi possível escrever no arquivo '{args.output_file}': {e}")

        print("\n--- Arquitetura Gerada (SANDL) ---")
        print(random_architecture)
        print("\n" + "-"*45)

    except FileNotFoundError:
        print(f"ERRO: O arquivo da gramática '{args.grammar_file}' não foi encontrado.")
    except Exception as e:
        print(f"ERRO: Ocorreu um erro inesperado: {e}")