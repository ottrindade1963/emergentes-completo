import os
import passo4_model_train_config as config
from passo4_model_train_processor import run_training_for_all
from passo4_model_train_visualizer import TrainingVisualizer

def run_model_training_pipeline():
    """
    Executa o pipeline completo de Treinamento de Modelos:
    1. Carrega os datasets processados no Passo 3
    2. Divide em Treino/Validação/Teste (Walk-forward)
    3. Treina os modelos (RF, XGBoost, SARIMAX)
    4. Salva os modelos em disco (.pkl)
    5. Gera visualizações de histórico de treino
    """
    print("="*50)
    print("INICIANDO PIPELINE DE TREINAMENTO DE MODELOS (PASSO 4)")
    print("="*50)
    
    # 1. Processamento: Treinar modelos
    print("\n[1/2] Treinando Modelos...")
    run_training_for_all()
    
    # 2. Visualizacao: Gerar graficos de treino
    print("\n[2/2] Gerando Visualizacoes de Treino...")
    try:
        visualizer = TrainingVisualizer()
        visualizer.plot_real_training_metrics()
        visualizer.plot_predictions_vs_actual()
        visualizer.plot_best_model_analysis()
        visualizer.plot_predictions_comparison()
        print("  -> Visualizacoes geradas com sucesso!")
    except Exception as e:
        print(f"  AVISO: Nao foi possivel gerar visualizacoes: {e}")
        print(f"  Os modelos foram treinados e salvos com sucesso.")
        print(f"  As visualizacoes podem ser geradas posteriormente no Passo 5.")
    
    print("\n" + "="*50)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print(f"Modelos salvos em: {config.OUTPUT_DIR}")
    print(f"Visualizações salvas em: {os.path.join(config.OUTPUT_DIR, 'visualizacoes_treino')}")
    print("="*50)

if __name__ == "__main__":
    run_model_training_pipeline()
