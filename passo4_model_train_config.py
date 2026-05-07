import os
import sys

# Detectar se está no Colab
IN_COLAB = 'google.colab' in sys.modules

# Detectar o diretório raiz dinamicamente
if IN_COLAB:
    import glob
    # Encontrar o diretório do repositório clonado
    dirs = [d for d in os.listdir('/content') if os.path.isdir(f'/content/{d}') and d not in ['.config', 'sample_data']]
    REPO_DIR = dirs[0] if dirs else os.getcwd()
    os.chdir(REPO_DIR)
    BASE_DIR = REPO_DIR
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import os

# Configurações de Caminhos
# Detecção automática de ambiente (Colab vs Local)
    BASE_DIR = os.getcwd()

DATA_DIR = os.path.join(BASE_DIR, 'dados_engenharia')  # Lê os dados gerados no Passo 3
OUTPUT_DIR = os.path.join(BASE_DIR, 'modelos_treinados')

# Garantir que o diretório de saída existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variável Alvo
TARGET_VAR = 'valor_agregado_industrial_percent_pib'

# Configurações de Divisão Temporal
# Dados cobrem 1996-2024 (~28 anos)
TRAIN_END_YEAR = 2016   # Treino: 1996-2016 (~21 anos = 75%)
VAL_END_YEAR = 2019     # Validação: 2017-2019 (3 anos = ~11%)
# Teste: 2020-2024 (5 anos = ~14%)

# Modelos a serem treinados
MODELS_TO_TRAIN = ['RandomForest', 'XGBoost', 'SARIMAX', 'LSTM', 'TFT']

# Datasets e Estratégias a processar
DATASETS = ['nao_agregado', 'inner', 'left', 'outer']
STRATEGIES = ['A1_Direta', 'A2_PCA', 'A3_Interacao']

# Seed para reprodutibilidade
RANDOM_STATE = 42
