# Arquivo: src/custom_operators.py
# Versão corrigida para criar os indivíduos corretamente.

import os
from algorithm.parameters import params
from representation.individual import Individual
# --- IMPORTAÇÃO CRÍTICA ---
# Precisamos da classe Tree para criar a árvore de derivação inicial.
from representation.tree import Tree
from utilities.representation.check_methods import check_ind

def initialisation_from_seed_files(size: int) -> list:
    """
    Operador de inicialização que lê fenótipos de arquivos .sandl de uma pasta
    especificada e os usa para criar a população inicial.
    """
    individuals = []
    
    # 1. Pega o NOME da subpasta de sementes dos parâmetros.
    seed_folder_name = params.get('TARGET_SEED_FOLDER')
    if not seed_folder_name:
        raise ValueError("A chave 'TARGET_SEED_FOLDER' não foi especificada nos parâmetros.")

    # 2. Constrói o caminho COMPLETO para a pasta de sementes.
    #    '..' sobe um nível (de 'src/' para a raiz 'PonyGE2/')
    #    'seeds' entra na pasta de sementes padrão.
    #    'seed_folder_name' é o nome da sua pasta específica.
    seed_folder_path = os.path.join("..", "seeds", seed_folder_name)
    
    # 3. Verifica se o caminho construído existe.
    if not os.path.isdir(seed_folder_path):
        raise FileNotFoundError(f"A pasta de sementes '{seed_folder_path}' não foi encontrada. Verifique o caminho e a sua estrutura de pastas.")
        
    print(f"INFO: Usando inicialização a partir de arquivos na pasta '{seed_folder_path}'.")
    
    # Garante que a gramática BNF esteja carregada
    if 'BNF_GRAMMAR' not in params:
        from representation.grammar import Grammar
        params['BNF_GRAMMAR'] = Grammar(params['GRAMMAR_FILE'])

    phenotype_files = [f for f in os.listdir(seed_folder_path) if f.endswith(".sandl")]
    
    if len(phenotype_files) > size:
        phenotype_files = phenotype_files[:size]

    for filename in phenotype_files:
        filepath = os.path.join(seed_folder_path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            phenotype_string = f.read()

        # --- A CORREÇÃO FINAL ESTÁ AQUI ---
        
        # 1. Crie uma árvore vazia. A raiz da árvore deve ser
        #    o símbolo inicial da sua gramática (geralmente '<ann>').
        start_symbol = params['BNF_GRAMMAR'].start_rule['symbol']
        print(f'Símbolo inicial: {start_symbol}')
        tree = Tree(start_symbol,None)

        # 2. Crie o indivíduo, passando a ÁRVORE VAZIA, não a gramática.
        new_ind = Individual(None, tree)
        
        print(f"Debug: {new_ind}")
        # 3. Atribua o fenótipo que você leu do arquivo.
        new_ind.phenotype = phenotype_string
        
        # 4. Chame o mapeamento reverso. `check_ind` irá usar o fenótipo para
        #    preencher a árvore e encontrar um genótipo válido.
        check_ind(new_ind, params['BNF_GRAMMAR'])
        
        # ------------------------------------
        
        if not new_ind.invalid:
            individuals.append(new_ind)
        else:
            print(f"AVISO: Mapeamento reverso falhou para o fenótipo do arquivo '{filename}'. Descartando.")

    if not individuals:
        raise RuntimeError("Nenhum fenótipo da pasta de sementes pôde ser mapeado com sucesso.")

    print(f"INFO: {len(individuals)} indivíduos carregados e mapeados com sucesso da pasta de sementes.")
    return individuals