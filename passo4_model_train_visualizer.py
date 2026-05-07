"""Visualizador de métricas de treinamento."""
import matplotlib
matplotlib.use('Agg')  # Backend não-interativo para Colab
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import passo4_model_train_config as config

class TrainingVisualizer:
    def __init__(self):
        self.output_dir = os.path.join(config.OUTPUT_DIR, 'visualizacoes_treino')
        # Garantir que o diretório pai existe
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

    def plot_training_history(self, model_name, dataset_name, strategy_name, history):
        """Visualiza o histórico de treinamento (loss) para modelos que suportam (ex: XGBoost, Redes Neurais)."""
        print(f"Gerando gráfico de histórico de treino para {model_name} ({dataset_name}-{strategy_name})...")
        
        plt.figure(figsize=(8, 5))
        
        if isinstance(history, dict) and 'validation_0' in history:
            # Formato XGBoost
            epochs = len(history['validation_0']['rmse'])
            x_axis = range(0, epochs)
            
            plt.plot(x_axis, history['validation_0']['rmse'], label='Train')
            if 'validation_1' in history:
                plt.plot(x_axis, history['validation_1']['rmse'], label='Validation')
                
            plt.title(f'Histórico de Treinamento - {model_name} ({dataset_name}-{strategy_name})')
            plt.xlabel('Épocas (Boosting Rounds)')
            plt.ylabel('RMSE')
            plt.legend()
            
            filename = f'history_{model_name}_{dataset_name}_{strategy_name}.png'
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
            plt.close()
        else:
            print(f"  -> Histórico não disponível ou formato não suportado para {model_name}.")

    def plot_real_training_metrics(self):
        """Gera gráficos reais baseados nos logs de treinamento salvos."""
        print("Gerando gráficos reais de treinamento...")
        
        log_path = os.path.join(config.OUTPUT_DIR, 'training_logs.pkl')
        if not os.path.exists(log_path):
            print(f"  -> Arquivo de logs não encontrado: {log_path}")
            return
            
        import joblib
        logs = joblib.load(log_path)
        
        if not logs:
            print("  -> Logs de treinamento vazios.")
            return
            
        # Extrair dados para DataFrame
        records = []
        for key, data in logs.items():
            # key formato: "Modelo_Dataset_Estrategia"
            parts = key.split('_', 1)
            if len(parts) == 2:
                model = parts[0]
                dataset_strat = parts[1]
                
                # Separar dataset e estrategia para análises mais ricas
                ds_parts = dataset_strat.split('_A')
                if len(ds_parts) == 2:
                    dataset = ds_parts[0]
                    strategy = 'A' + ds_parts[1]
                else:
                    dataset = dataset_strat
                    strategy = 'N/A'
                
                records.append({
                    'Modelo': model,
                    'Cenario': dataset_strat,
                    'Dataset': dataset,
                    'Estrategia': strategy,
                    'RMSE_Treino': data.get('train_rmse', 0),
                    'R2_Treino': data.get('train_r2', 0),
                    'Tempo_s': data.get('train_time', 0)
                })
                
        if not records:
            return
            
        df = pd.DataFrame(records)
        
        # 1. Gráfico de Tempo de Treinamento Real
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Modelo', y='Tempo_s', hue='Cenario', data=df)
        plt.title('Tempo de Treinamento Real por Modelo e Cenário', fontsize=14)
        plt.ylabel('Tempo (segundos)')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tempo_treino_real.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gráfico de R2 no Treino
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Modelo', y='R2_Treino', hue='Cenario', data=df)
        plt.title('R² no Conjunto de Treinamento', fontsize=14)
        plt.ylabel('R² Score')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'r2_treino_real.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Gráfico de RMSE no Treino
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Modelo', y='RMSE_Treino', hue='Cenario', data=df)
        plt.title('RMSE no Conjunto de Treinamento', fontsize=14)
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rmse_treino_real.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Heatmap de R2 por Modelo e Cenário
        plt.figure(figsize=(14, 8))
        pivot_r2 = df.pivot(index='Modelo', columns='Cenario', values='R2_Treino')
        sns.heatmap(pivot_r2, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
        plt.title('Heatmap de R² no Treinamento por Modelo e Cenário', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'heatmap_r2_treino.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Boxplot de RMSE por Modelo
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Modelo', y='RMSE_Treino', data=df, palette='Set3')
        plt.title('Distribuição do RMSE de Treinamento por Modelo', fontsize=14)
        plt.ylabel('RMSE')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'boxplot_rmse_treino.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Comparação de Datasets (Média de RMSE)
        plt.figure(figsize=(10, 6))
        avg_rmse_ds = df.groupby('Dataset')['RMSE_Treino'].mean().reset_index()
        sns.barplot(x='Dataset', y='RMSE_Treino', data=avg_rmse_ds, palette='coolwarm')
        plt.title('RMSE Médio de Treinamento por Dataset', fontsize=14)
        plt.ylabel('RMSE Médio')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rmse_medio_por_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  -> 6 Gráficos reais gerados em {self.output_dir}")

    def plot_predictions_vs_actual(self):
        """Gera gráficos de previsões vs valores reais para cada modelo."""
        print("Gerando gráficos de previsões vs valores reais...")
        
        log_path = os.path.join(config.OUTPUT_DIR, 'training_logs.pkl')
        if not os.path.exists(log_path):
            print("  -> Arquivo de logs não encontrado.")
            return
            
        import joblib
        logs = joblib.load(log_path)
        
        # Extrair dados para ranking
        records = []
        for key, data in logs.items():
            parts = key.split('_', 1)
            if len(parts) == 2:
                model = parts[0]
                dataset_strat = parts[1]
                
                records.append({
                    'Modelo': model,
                    'Cenario': dataset_strat,
                    'R2_Val': data.get('val_r2', 0),
                    'RMSE_Val': data.get('val_rmse', 0),
                    'MAE_Val': data.get('val_mae', 0)
                })
        
        if not records:
            return
            
        df = pd.DataFrame(records)
        
        # 1. Ranking de Modelos por R2 (Validação)
        plt.figure(figsize=(12, 6))
        ranking_r2 = df.groupby('Modelo')['R2_Val'].mean().sort_values(ascending=False)
        colors = ['green' if x > 0.7 else 'orange' if x > 0.5 else 'red' for x in ranking_r2.values]
        sns.barplot(x=ranking_r2.index, y=ranking_r2.values, palette=colors)
        plt.title('Ranking de Modelos por R² em Validação (Melhor Previsão)', fontsize=14, fontweight='bold')
        plt.ylabel('R² Médio em Validação')
        plt.ylim(0, 1.0)
        for i, v in enumerate(ranking_r2.values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_ranking_modelos_r2_validacao.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Ranking de Modelos por RMSE (menor é melhor)
        plt.figure(figsize=(12, 6))
        ranking_rmse = df.groupby('Modelo')['RMSE_Val'].mean().sort_values()
        colors = ['green' if x < 5 else 'orange' if x < 7 else 'red' for x in ranking_rmse.values]
        sns.barplot(x=ranking_rmse.index, y=ranking_rmse.values, palette=colors)
        plt.title('Ranking de Modelos por RMSE em Validação (Menor é Melhor)', fontsize=14, fontweight='bold')
        plt.ylabel('RMSE Médio em Validação')
        for i, v in enumerate(ranking_rmse.values):
            plt.text(i, v + 0.1, f'{v:.3f}', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_ranking_modelos_rmse_validacao.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scatter: R2 vs RMSE (melhor modelo no canto superior esquerdo)
        plt.figure(figsize=(11, 8))
        for modelo in df['Modelo'].unique():
            subset = df[df['Modelo'] == modelo]
            plt.scatter(subset['RMSE_Val'].mean(), subset['R2_Val'].mean(), s=300, label=modelo, alpha=0.7, edgecolors='black', linewidth=2)
            plt.text(subset['RMSE_Val'].mean() + 0.15, subset['R2_Val'].mean() + 0.02, modelo, fontsize=11, fontweight='bold')
        
        plt.title('Trade-off: RMSE vs R² em Validação\n(Melhor Modelo no Canto Superior Esquerdo)', fontsize=14, fontweight='bold')
        plt.xlabel('RMSE Médio em Validação (Menor é Melhor)', fontsize=12)
        plt.ylabel('R² Médio em Validação (Maior é Melhor)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='R² = 0.7 (Bom)')
        plt.axvline(x=5, color='orange', linestyle='--', alpha=0.3, label='RMSE = 5 (Aceitável)')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_scatter_r2_vs_rmse_validacao.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Boxplot: Distribuição de R2 por Modelo
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Modelo', y='R2_Val', data=df, palette='Set2')
        plt.title('Distribuição de R² por Modelo em Validação (Todos os 12 Cenários)', fontsize=14, fontweight='bold')
        plt.ylabel('R² em Validação')
        plt.ylim(-0.2, 1.0)
        plt.axhline(y=0.7, color='green', linestyle='--', alpha=0.3, label='Limite Bom')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_boxplot_r2_validacao.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Heatmap: R2 por Modelo e Cenário (Todos os 60 modelos)
        plt.figure(figsize=(16, 7))
        pivot_r2 = df.pivot(index='Modelo', columns='Cenario', values='R2_Val')
        sns.heatmap(pivot_r2, annot=True, cmap='RdYlGn', fmt='.3f', linewidths=.5, vmin=0, vmax=1, cbar_kws={'label': 'R² em Validação'})
        plt.title('Heatmap de R² por Modelo e Cenário em Validação (Todos os 60 Modelos)', fontsize=14, fontweight='bold')
        plt.xlabel('Cenário (Dataset + Estratégia)')
        plt.ylabel('Modelo')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_heatmap_r2_completo_validacao.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Comparação MAE por Modelo
        plt.figure(figsize=(12, 6))
        ranking_mae = df.groupby('Modelo')['MAE_Val'].mean().sort_values()
        sns.barplot(x=ranking_mae.index, y=ranking_mae.values, palette='viridis')
        plt.title('Ranking de Modelos por MAE em Validação (Menor é Melhor)', fontsize=14, fontweight='bold')
        plt.ylabel('MAE Médio em Validação')
        for i, v in enumerate(ranking_mae.values):
            plt.text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '06_ranking_modelos_mae_validacao.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  -> 6 Gráficos de Previsão gerados em {self.output_dir}")

    def plot_best_model_analysis(self):
        """Analisa qual modelo, estratégia e cenário faz a melhor previsão."""
        print("Gerando análise detalhada do melhor modelo...")
        
        log_path = os.path.join(config.OUTPUT_DIR, 'training_logs.pkl')
        if not os.path.exists(log_path):
            print("  -> Arquivo de logs não encontrado.")
            return
            
        import joblib
        logs = joblib.load(log_path)
        
        # Extrair dados com parsing completo
        records = []
        for key, data in logs.items():
            parts = key.split('_', 1)
            if len(parts) == 2:
                model = parts[0]
                dataset_strat = parts[1]
                
                # Separar dataset e estratégia
                if '_A' in dataset_strat:
                    ds_parts = dataset_strat.rsplit('_A', 1)
                    dataset = ds_parts[0]
                    estrategia = 'A' + ds_parts[1]
                else:
                    dataset = dataset_strat
                    estrategia = 'N/A'
                
                records.append({
                    'Modelo': model,
                    'Dataset': dataset,
                    'Estrategia': estrategia,
                    'Cenario': dataset_strat,
                    'R2_Val': data.get('val_r2', 0),
                    'RMSE_Val': data.get('val_rmse', 0),
                    'MAE_Val': data.get('val_mae', 0)
                })
        
        if not records:
            return
            
        df = pd.DataFrame(records)
        
        # 1. Identificar o MELHOR modelo geral (por R2)
        best_idx = df['R2_Val'].idxmax()
        best_row = df.loc[best_idx]
        
        # 2. Gráfico: Top 10 Melhores Combinações (Modelo + Estratégia + Dataset)
        plt.figure(figsize=(14, 8))
        top_10 = df.nlargest(10, 'R2_Val')[['Modelo', 'Estrategia', 'Dataset', 'R2_Val']].copy()
        top_10['Combinacao'] = top_10['Modelo'] + '\n' + top_10['Estrategia'] + '\n' + top_10['Dataset']
        
        colors = ['gold' if i == 0 else 'silver' if i == 1 else 'chocolate' if i == 2 else 'steelblue' for i in range(len(top_10))]
        bars = plt.barh(range(len(top_10)), top_10['R2_Val'].values, color=colors)
        plt.yticks(range(len(top_10)), top_10['Combinacao'].values, fontsize=10)
        plt.xlabel('R² em Validação', fontsize=12, fontweight='bold')
        plt.title('Top 10 Melhores Combinações: Modelo + Estratégia + Dataset', fontsize=14, fontweight='bold')
        plt.xlim(0, 1.0)
        
        # Adicionar valores nas barras
        for i, (bar, val) in enumerate(zip(bars, top_10['R2_Val'].values)):
            plt.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
                    va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '07_top10_melhores_combinacoes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Heatmap: Modelo vs Estratégia (agregado por dataset)
        plt.figure(figsize=(10, 7))
        pivot_model_strat = df.pivot_table(index='Modelo', columns='Estrategia', values='R2_Val', aggfunc='mean')
        sns.heatmap(pivot_model_strat, annot=True, cmap='RdYlGn', fmt='.4f', linewidths=1, vmin=0, vmax=1, 
                   cbar_kws={'label': 'R² Médio em Validação'}, linecolor='black')
        plt.title('Heatmap: Melhor Modelo x Melhor Estratégia\n(Agregado por Dataset)', fontsize=14, fontweight='bold')
        plt.xlabel('Estratégia de Agregação', fontsize=12, fontweight='bold')
        plt.ylabel('Modelo', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '08_heatmap_modelo_vs_estrategia.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Heatmap: Modelo vs Dataset (agregado por estratégia)
        plt.figure(figsize=(10, 7))
        pivot_model_ds = df.pivot_table(index='Modelo', columns='Dataset', values='R2_Val', aggfunc='mean')
        sns.heatmap(pivot_model_ds, annot=True, cmap='RdYlGn', fmt='.4f', linewidths=1, vmin=0, vmax=1,
                   cbar_kws={'label': 'R² Médio em Validação'}, linecolor='black')
        plt.title('Heatmap: Melhor Modelo x Melhor Dataset\n(Agregado por Estratégia)', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('Modelo', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '09_heatmap_modelo_vs_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Heatmap: Estratégia vs Dataset (agregado por modelo)
        plt.figure(figsize=(10, 7))
        pivot_strat_ds = df.pivot_table(index='Estrategia', columns='Dataset', values='R2_Val', aggfunc='mean')
        sns.heatmap(pivot_strat_ds, annot=True, cmap='RdYlGn', fmt='.4f', linewidths=1, vmin=0, vmax=1,
                   cbar_kws={'label': 'R² Médio em Validação'}, linecolor='black')
        plt.title('Heatmap: Melhor Estratégia x Melhor Dataset\n(Agregado por Modelo)', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12, fontweight='bold')
        plt.ylabel('Estratégia', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '10_heatmap_estrategia_vs_dataset.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Gráfico de Linha: Performance por Estratégia (um modelo por linha)
        plt.figure(figsize=(14, 8))
        for modelo in df['Modelo'].unique():
            subset = df[df['Modelo'] == modelo].groupby('Estrategia')['R2_Val'].mean().reset_index()
            plt.plot(subset['Estrategia'], subset['R2_Val'], marker='o', linewidth=2.5, markersize=8, label=modelo)
        
        plt.title('Performance por Estratégia de Agregação (Todos os Modelos)', fontsize=14, fontweight='bold')
        plt.xlabel('Estratégia de Agregação', fontsize=12, fontweight='bold')
        plt.ylabel('R² Médio em Validação', fontsize=12, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '11_performance_por_estrategia.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Imprimir resumo no console
        print(f"\n  *** MELHOR MODELO IDENTIFICADO ***")
        print(f"  Modelo: {best_row['Modelo']}")
        print(f"  Estratégia: {best_row['Estrategia']}")
        print(f"  Dataset: {best_row['Dataset']}")
        print(f"  Cenário Completo: {best_row['Cenario']}")
        print(f"  R² em Validação: {best_row['R2_Val']:.4f}")
        print(f"  RMSE em Validação: {best_row['RMSE_Val']:.4f}")
        print(f"  MAE em Validação: {best_row['MAE_Val']:.4f}")
        print(f"  *** FIM DO RESUMO ***\n")
        
        print(f"  -> 5 Gráficos de Análise Detalhada gerados em {self.output_dir}")

    def plot_predictions_comparison(self):
        """Gera gráficos comparando previsões dos modelos vs valores reais."""
        print("Gerando gráficos de previsões vs valores reais...")
        
        import glob
        model_files = glob.glob(os.path.join(config.OUTPUT_DIR, '*.pkl'))
        
        if not model_files:
            print("  -> Nenhum modelo encontrado.")
            return
        
        # Carregar dados de previsão de todos os modelos
        predictions_data = {}
        for model_file in model_files:
            try:
                model_data = pd.read_pickle(model_file)
                if 'y_val' in model_data and 'model' in model_data:
                    model_obj = model_data['model']
                    y_val = model_data.get('y_val', [])
                    X_val = model_data.get('X_val', [])
                    
                    if len(X_val) > 0 and len(y_val) > 0:
                        # Fazer previsão
                        try:
                            y_pred = model_obj.predict(X_val)
                            filename = os.path.basename(model_file).replace('.pkl', '')
                            predictions_data[filename] = {
                                'y_real': y_val,
                                'y_pred': y_pred
                            }
                        except:
                            pass
            except:
                pass
        
        if not predictions_data:
            print("  -> Não foi possível carregar previsões.")
            return
        
        # 1. Scatter: Previsões vs Valores Reais (todos os modelos)
        plt.figure(figsize=(14, 10))
        colors = plt.cm.tab20(np.linspace(0, 1, len(predictions_data)))
        
        for idx, (model_name, data) in enumerate(predictions_data.items()):
            y_real = data['y_real']
            y_pred = data['y_pred']
            plt.scatter(y_real, y_pred, alpha=0.6, s=50, label=model_name, color=colors[idx])
        
        # Linha de referência (previsão perfeita)
        min_val = min([data['y_real'].min() for data in predictions_data.values()])
        max_val = max([data['y_real'].max() for data in predictions_data.values()])
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Previsão Perfeita')
        
        plt.xlabel('Valores Reais', fontsize=12, fontweight='bold')
        plt.ylabel('Valores Preditos', fontsize=12, fontweight='bold')
        plt.title('Previsões vs Valores Reais - Todos os Modelos', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '12_scatter_predicoes_vs_reais_todos.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Gráfico de Linha: Previsões vs Valores Reais (melhor modelo)
        best_model = max(predictions_data.items(), key=lambda x: r2_score(x[1]['y_real'], x[1]['y_pred']))
        best_model_name = best_model[0]
        best_y_real = best_model[1]['y_real']
        best_y_pred = best_model[1]['y_pred']
        
        plt.figure(figsize=(14, 6))
        x_axis = range(len(best_y_real))
        plt.plot(x_axis, best_y_real, 'o-', linewidth=2, markersize=6, label='Valores Reais', color='blue')
        plt.plot(x_axis, best_y_pred, 's--', linewidth=2, markersize=5, label='Valores Preditos', color='red')
        plt.fill_between(x_axis, best_y_real, best_y_pred, alpha=0.2, color='gray')
        
        plt.xlabel('Observação', fontsize=12, fontweight='bold')
        plt.ylabel('Valor Agregado Industrial (% PIB)', fontsize=12, fontweight='bold')
        plt.title(f'Melhor Modelo: {best_model_name}\nPrevisões vs Valores Reais', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '13_linha_melhor_modelo_predicoes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Residuais (Erros) do Melhor Modelo
        residuais = best_y_real - best_y_pred
        
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(best_y_pred, residuais, alpha=0.6, s=80, color='purple')
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Valores Preditos', fontsize=11, fontweight='bold')
        plt.ylabel('Resíduos (Reais - Preditos)', fontsize=11, fontweight='bold')
        plt.title(f'Resíduos do Melhor Modelo: {best_model_name}', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.hist(residuais, bins=15, color='green', alpha=0.7, edgecolor='black')
        plt.xlabel('Resíuo', fontsize=11, fontweight='bold')
        plt.ylabel('Frequência', fontsize=11, fontweight='bold')
        plt.title('Distribuição dos Resíduos', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '14_residuais_melhor_modelo.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Comparação de Erro Absoluto por Modelo
        mae_by_model = {}
        for model_name, data in predictions_data.items():
            mae = mean_absolute_error(data['y_real'], data['y_pred'])
            mae_by_model[model_name] = mae
        
        mae_sorted = dict(sorted(mae_by_model.items(), key=lambda x: x[1]))
        
        plt.figure(figsize=(14, 6))
        colors_mae = ['green' if v < 3 else 'orange' if v < 5 else 'red' for v in mae_sorted.values()]
        bars = plt.barh(list(mae_sorted.keys()), list(mae_sorted.values()), color=colors_mae)
        plt.xlabel('Erro Absoluto Médio (MAE)', fontsize=12, fontweight='bold')
        plt.title('Comparação de MAE entre Modelos\n(Menor é Melhor)', fontsize=14, fontweight='bold')
        
        for i, (bar, val) in enumerate(zip(bars, mae_sorted.values())):
            plt.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                    va='center', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '15_mae_comparacao_modelos.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Heatmap: Previsões vs Valores Reais (top 5 modelos)
        top_5_models = sorted(predictions_data.items(), 
                            key=lambda x: r2_score(x[1]['y_real'], x[1]['y_pred']), 
                            reverse=True)[:5]
        
        # Criar matriz com previsões
        comparison_matrix = []
        model_labels = []
        for model_name, data in top_5_models:
            comparison_matrix.append(data['y_pred'])
            model_labels.append(model_name)
        
        comparison_matrix.append(top_5_models[0][1]['y_real'])  # Adicionar valores reais
        model_labels.append('VALORES REAIS')
        
        comparison_df = pd.DataFrame(comparison_matrix, index=model_labels)
        
        plt.figure(figsize=(16, 8))
        sns.heatmap(comparison_df, annot=True, fmt='.2f', cmap='RdYlGn', cbar_kws={'label': 'Valor Agregado Industrial (% PIB)'},
                   linewidths=0.5, linecolor='black')
        plt.title('Heatmap: Previsões dos Top 5 Modelos vs Valores Reais', fontsize=14, fontweight='bold')
        plt.xlabel('Observação', fontsize=12, fontweight='bold')
        plt.ylabel('Modelo / Valores Reais', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '16_heatmap_top5_predicoes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  -> 5 Gráficos de Previsões vs Valores Reais gerados em {self.output_dir}")

if __name__ == "__main__":
    viz = TrainingVisualizer()
    viz.plot_real_training_metrics()
    viz.plot_predictions_vs_actual()
    viz.plot_best_model_analysis()
    viz.plot_predictions_comparison()
