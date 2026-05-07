import os
import passo3_feat_eng_config as config
from passo3_feat_eng_processor import load_and_process_datasets
from passo3_feat_eng_visualizer import FeatureVisualizer

def run_feature_engineering_pipeline():
    """
    Executa o pipeline completo de Engenharia de Features:
    1. Carrega os datasets agregados (Inner, Left, Outer)
    2. Aplica as 3 estratégias (A1, A2, A3)
    3. Salva os novos datasets
    4. Gera visualizações (Heatmaps, PCA Variance)
    """
    print("="*50)
    print("INICIANDO PIPELINE DE ENGENHARIA DE FEATURES (PASSO 3)")
    print("="*50)
    
    # 1. Processamento: Carregar e aplicar estratégias
    print("\n[1/2] Processando Datasets e Aplicando Estratégias...")
    datasets_dict = load_and_process_datasets()
    
    # 2. Visualização: Gerar gráficos
    print("\n[2/2] Gerando Visualizações...")
    visualizer = FeatureVisualizer(datasets_dict)
    visualizer.generate_all_visualizations()
    
    print("\n" + "="*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Datasets salvos em: {config.OUTPUT_DIR}")
    print(f"Visualizações salvas em: {os.path.join(config.OUTPUT_DIR, 'visualizacoes')}")
    print("="*50)

if __name__ == "__main__":
    run_feature_engineering_pipeline()
