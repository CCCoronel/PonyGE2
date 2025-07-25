#!/bin/bash

# --- CONFIGURAÇÃO ---
# Ajuste estes caminhos para corresponder à sua estrutura de pastas
GRAMMAR_FILE="grammars/renewable_energy/sandl-gen.bnf"
OUTPUT_DIR="src/seeds/sandl_architectures" # Diretório onde as sementes serão salvas
POPULATION_SIZE=50
GENERATOR_SCRIPT="src/generatorSandl/sandl_generator.py" # Caminho para o seu gerador

# --- EXECUÇÃO ---
echo "Criando o diretório de sementes em '$OUTPUT_DIR'..."
mkdir -p $OUTPUT_DIR

echo "Iniciando a geração de $POPULATION_SIZE fenótipos iniciais..."
for i in $(seq 1 $POPULATION_SIZE); do
    echo "Gerando indivíduo $i..."
    python $GENERATOR_SCRIPT $GRAMMAR_FILE --output-file "$OUTPUT_DIR/arch_$i.sandl" --seed $i
done

echo "Geração de fenótipos iniciais concluída."