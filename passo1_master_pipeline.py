"""Pipeline Mestre do Passo 1: Extração de Dados Reais."""
import os
import sys

def executar_passo1_completo():
    print("=" * 70)
    print("INICIANDO PASSO 1 COMPLETO: EXTRAÇÃO DE DADOS REAIS")
    print("=" * 70)
    
    # 1. Extração Quantitativa (WDI)
    print("\n[1/2] EXECUTANDO EXTRAÇÃO DE DADOS QUANTITATIVOS (WDI)...")
    from passo1_extracao_pipeline import executar
    executar()
    
    # 2. Extração Qualitativa (WGI)
    print("\n[2/2] EXECUTANDO EXTRAÇÃO DE DADOS QUALITATIVOS (WGI)...")
    from passo1_extracao_quali_processor import executar_extracao_wgi
    executar_extracao_wgi()
    
    print("\n" + "=" * 70)
    print("PASSO 1 CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

if __name__ == "__main__":
    executar_passo1_completo()
