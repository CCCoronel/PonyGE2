import argparse
import random
import re
from typing import Dict, List, Any

class SANDLGenerator:
    """
    Gera arquiteturas de redes neurais aleatórias baseadas em uma gramática BNF
    preparada para Evolução Gramatical.
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

    def _expand(self, symbol: str, depth: int, max_depth: int) -> str:
        """Expande recursivamente um símbolo da gramática, controlando a profundidade."""
        if symbol not in self.grammar:
            return symbol.strip('"')

        is_recursive = any(symbol in p for p in self.grammar[symbol])

        if is_recursive and depth >= max_depth:
            non_recursive_productions = [p for p in self.grammar[symbol] if symbol not in p]
            if not non_recursive_productions:
                return ""
            production = random.choice(non_recursive_productions)
        else:
            production = random.choice(self.grammar[symbol])

        if production == ['<<empty>>']:
            return ""

        parts = []
        for part_symbol in production:
            new_depth = depth + 1 if part_symbol == symbol else depth
            
            # Lógica de formatação para legibilidade
            if part_symbol == '"{"':
                self.indent_level += 1
                parts.append(" {\n" + self._get_indent())
            elif part_symbol == '"}"':
                self.indent_level -= 1
                if parts and parts[-1].strip() == "": parts.pop()
                parts.append("\n" + self._get_indent() + "}")
            elif part_symbol == '";"':
                if parts and parts[-1].endswith(" "):
                    parts[-1] = parts[-1].strip()
                parts.append(";\n" + self._get_indent())
            else:
                expanded_part = self._expand(part_symbol, new_depth, max_depth)
                if expanded_part:
                    parts.append(expanded_part + " ")

        return "".join(parts)

    def _get_indent(self) -> str:
        return "    " * self.indent_level

    def generate(self, max_depth: int = 15) -> str:
        """Gera e formata uma definição de arquitetura SANDL completa."""
        if not self.grammar:
            return "Erro: A gramática não foi carregada corretamente."

        self.indent_level = 0
        raw_output = self._expand('<ann>', 0, max_depth)
        
        # Pós-processamento para limpar a formatação
        clean_output = re.sub(r'\s+([;{}])', r'\1', raw_output)
        clean_output = re.sub(r'(\w)({)', r'\1 \2', clean_output)
        clean_output = "\n".join(line for line in clean_output.split('\n') if line.strip())

        return clean_output


# --- Para Executar o Programa ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gera uma arquitetura SANDL aleatória e a salva em um arquivo."
    )
    parser.add_argument(
        "grammar_file",
        type=str,
        help="O caminho para o arquivo .bnf contendo a gramática SANDL."
    )
    # --- ALTERAÇÃO 1: Adicionado argumento para o arquivo de saída ---
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

        # Gera a arquitetura
        random_architecture = generator.generate(max_depth=args.max_depth)
        
        # --- ALTERAÇÃO 2: Bloco para salvar o arquivo ---
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(random_architecture)
            print(f"INFO: Arquitetura salva com sucesso em '{args.output_file}'")
        except IOError as e:
            print(f"ERRO: Não foi possível escrever no arquivo '{args.output_file}': {e}")
        # --- Fim da Alteração 2 ---

        # Imprime a saída no terminal como antes
        print("\n--- Arquitetura Gerada (SANDL) ---")
        print(random_architecture)
        print("\n" + "-"*45)

    except FileNotFoundError:
        print(f"ERRO: O arquivo da gramática '{args.grammar_file}' não foi encontrado.")
    except Exception as e:
        print(f"ERRO: Ocorreu um erro inesperado: {e}")