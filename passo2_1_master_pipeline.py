"""Pipeline Mestre do Passo 2.1: Limpeza, Agregação e EDA dos Agregados."""
import os
import sys
import pandas as pd

def executar_passo2_1_completo():
    print("=" * 70)
    print("INICIANDO PASSO 2.1 COMPLETO: LIMPEZA, AGREGAÇÃO E EDA")
    print("=" * 70)
    
    # 1. Limpeza
    print("\n[1/4] EXECUTANDO LIMPEZA DE DADOS QUANTITATIVOS...")
    from passo2_1_limpeza_pipeline import executar_limpeza
    df_wdi_limpo, _ = executar_limpeza()
    
    # 2. Agregação
    print("\n[2/4] EXECUTANDO AGREGAÇÃO (3 MÉTODOS)...")
    from passo2_1_agregacao_pipeline import executar_agregacao
    executar_agregacao()
    
    # 3. EDA Dados Não Agregados Limpos (WDI + WGI)
    print("\n[3/4] EXECUTANDO ANÁLISE EXPLORATÓRIA DOS DADOS NÃO AGREGADOS LIMPOS...")
    try:
        # Carregar dados qualitativos limpos
        df_wgi_limpo = pd.read_csv('dados_qualitativos.csv')
        
        # Executar EDA dos dados não agregados
        from passo2_1_eda_nao_agreg_visualizer import executar_eda_nao_agregado
        executar_eda_nao_agregado(df_wdi_limpo, df_wgi_limpo)
    except Exception as e:
        print(f"  AVISO: Não foi possível executar EDA dos não agregados: {e}")
        print("  Continuando com EDA dos agregados...")
    
    # 4. EDA Agregados
    print("\n[4/4] EXECUTANDO ANÁLISE EXPLORATÓRIA DOS AGREGADOS...")
    from passo2_1_eda_agreg_pipeline import executar_eda_agregados
    executar_eda_agregados()
    
    print("\n" + "=" * 70)
    print("PASSO 2.1 CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    executar_passo2_1_completo()
