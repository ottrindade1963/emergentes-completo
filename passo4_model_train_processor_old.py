import os
import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

import passo4_model_train_config as config


class ModelTrainer:
    """
    Treinador robusto de modelos com:
    - Divisão temporal correta (treino/validação/teste por ano)
    - TimeSeriesSplit para validação cruzada
    - GridSearchCV/RandomizedSearchCV com grids extensos
    - SARIMAX real com features consistentes
    - LSTM proxy (MLPRegressor com tuning) e TFT proxy (GradientBoosting com tuning)
    """
    
    def __init__(self, df, dataset_name, strategy_name):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name
        self.models = {}
        self.training_metrics = {}
        self.scaler = StandardScaler()
        self.sarimax_features = None  # Features usadas pelo SARIMAX
        
    def prepare_data(self):
        """Prepara os dados com divisão temporal correta."""
        # Detectar coluna de ano (pode ser 'ano' ou 'year')
        if 'ano' in self.df.columns:
            self.year_col = 'ano'
        elif 'year' in self.df.columns:
            self.year_col = 'year'
        else:
            self.year_col = None
            
        # Colunas de identificação a remover das features
        id_cols = ['pais', 'country', 'country_code', 'codigo_iso3', 'fonte_dados']
        id_cols = [c for c in id_cols if c in self.df.columns]
        
        # Separar target
        if config.TARGET_VAR not in self.df.columns:
            raise ValueError(f"Target '{config.TARGET_VAR}' não encontrado no dataset.")
        
        # Remover linhas sem target
        valid_mask = self.df[config.TARGET_VAR].notna()
        df_valid = self.df[valid_mask].copy()
        
        # Separar y
        self.y = df_valid[config.TARGET_VAR].values
        
        # Separar X (remover target + id_cols, mas MANTER coluna de ano como feature)
        cols_to_drop_from_X = id_cols + [config.TARGET_VAR]
        self.X = df_valid.drop(columns=[c for c in cols_to_drop_from_X if c in df_valid.columns])
        
        # Manter apenas colunas numéricas
        self.X = self.X.select_dtypes(include=[np.number])
        
        # Tratar NaNs
        self.X = self.X.fillna(self.X.median())
        
        # Guardar nomes das features ANTES de escalar
        self.feature_names = list(self.X.columns)
        
        # Divisão temporal
        if self.year_col and self.year_col in self.X.columns:
            anos = self.X[self.year_col].values
            
            # Treino: até 2014, Validação: 2015-2017, Teste: 2018+
            train_mask = anos <= config.TRAIN_END_YEAR
            val_mask = (anos > config.TRAIN_END_YEAR) & (anos <= config.VAL_END_YEAR)
            test_mask = anos > config.VAL_END_YEAR
            
            self.X_train_raw = self.X[train_mask].copy()
            self.y_train = self.y[train_mask]
            self.X_val_raw = self.X[val_mask].copy()
            self.y_val = self.y[val_mask]
            self.X_test_raw = self.X[test_mask].copy()
            self.y_test = self.y[test_mask]
        else:
            # Fallback: divisão sequencial (80/10/10) respeitando ordem temporal
            n = len(self.X)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            
            self.X_train_raw = self.X.iloc[:train_end].copy()
            self.y_train = self.y[:train_end]
            self.X_val_raw = self.X.iloc[train_end:val_end].copy()
            self.y_val = self.y[train_end:val_end]
            self.X_test_raw = self.X.iloc[val_end:].copy()
            self.y_test = self.y[val_end:]
        
        # Guardar dados brutos para SARIMAX (nao escalados)
        self.X_train_unscaled = self.X_train_raw.copy()
        self.X_val_unscaled = self.X_val_raw.copy()
        self.X_test_unscaled = self.X_test_raw.copy()
        
        # Escalar features para RF, XGBoost, LSTM, TFT (fit no treino, transform em val/test)
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(self.X_train_raw),
            columns=self.feature_names,
            index=self.X_train_raw.index
        )
        self.X_val = pd.DataFrame(
            self.scaler.transform(self.X_val_raw),
            columns=self.feature_names,
            index=self.X_val_raw.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(self.X_test_raw),
            columns=self.feature_names,
            index=self.X_test_raw.index
        )
        
        print(f"  Dados preparados: Treino={len(self.X_train)}, Val={len(self.X_val)}, Teste={len(self.X_test)}")
        print(f"  Features: {len(self.feature_names)} | Target: {config.TARGET_VAR}")
        
    def train_random_forest(self):
        """Random Forest com GridSearchCV extenso + TimeSeriesSplit."""
        print("  -> Treinando Random Forest com GridSearchCV extenso...")
        
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # TimeSeriesSplit para validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=5)
        
        rf = RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        self.models['RandomForest'] = best_model
        
        # Métricas
        val_preds = best_model.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        val_r2 = r2_score(self.y_val, val_preds)
        
        print(f"     Melhores parâmetros: {grid_search.best_params_}")
        print(f"     Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
        print(f"     CV scores (neg_MSE): {grid_search.best_score_:.4f}")
        
        self.training_metrics['RandomForest'] = {
            'best_params': grid_search.best_params_,
            'cv_best_score': grid_search.best_score_,
            'val_rmse': val_rmse,
            'val_r2': val_r2
        }

    def train_xgboost(self):
        """XGBoost com RandomizedSearchCV extenso + TimeSeriesSplit."""
        print("  -> Treinando XGBoost com RandomizedSearchCV extenso...")
        
        param_dist = {
            'n_estimators': [100, 200, 300, 500, 800],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9, 11],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.5],
            'reg_alpha': [0, 0.01, 0.1, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0, 5.0]
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        xgb = XGBRegressor(
            random_state=config.RANDOM_STATE,
            objective='reg:squarederror',
            tree_method='hist'
        )
        
        random_search = RandomizedSearchCV(
            xgb, param_distributions=param_dist,
            n_iter=50,  # 50 combinações aleatórias
            cv=tscv,
            scoring='neg_mean_squared_error',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        self.models['XGBoost'] = best_model
        
        # Métricas
        val_preds = best_model.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        val_r2 = r2_score(self.y_val, val_preds)
        
        print(f"     Melhores parâmetros: {random_search.best_params_}")
        print(f"     Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
        print(f"     CV scores (neg_MSE): {random_search.best_score_:.4f}")
        
        self.training_metrics['XGBoost'] = {
            'best_params': random_search.best_params_,
            'cv_best_score': random_search.best_score_,
            'val_rmse': val_rmse,
            'val_r2': val_r2
        }

    def train_sarimax(self):
        """SARIMAX real com seleção consistente de features exógenas."""
        print("  -> Treinando SARIMAX real...")
        try:
            # Selecionar top N features mais correlacionadas com target
            n_exog = min(5, len(self.feature_names))
            
            # Calcular correlacoes no conjunto de treino (usando dados brutos)
            X_train_with_y = self.X_train_unscaled.copy()
            X_train_with_y['__target__'] = self.y_train
            correlations = X_train_with_y.corr()['__target__'].drop('__target__').abs().sort_values(ascending=False)
            
            # Selecionar top features (excluir coluna de ano se presente)
            top_features = []
            for feat in correlations.index:
                if feat not in [self.year_col, 'ano', 'year']:
                    top_features.append(feat)
                if len(top_features) >= n_exog:
                    break
            
            self.sarimax_features = top_features
            print(f"     Features exogenas selecionadas: {top_features}")
            
            # Treinar SARIMAX com dados brutos (nao escalados)
            model = SARIMAX(
                endog=self.y_train,
                exog=self.X_train_unscaled[top_features].values,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 0, 4),  # Sazonalidade de 4 períodos
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False, maxiter=500)
            
            # Guardar as features usadas para predição futura
            fitted_model._exog_features = top_features
            fitted_model._n_exog = len(top_features)
            self.models['SARIMAX'] = fitted_model
            
            # Métricas no conjunto de validação (usando dados brutos)
            try:
                val_preds = fitted_model.forecast(
                    steps=len(self.y_val),
                    exog=self.X_val_unscaled[top_features].values
                )
                val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
                val_r2 = r2_score(self.y_val, val_preds)
                print(f"     Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
                self.training_metrics['SARIMAX'] = {
                    'exog_features': top_features,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
            except Exception as e:
                print(f"     Aviso na validação SARIMAX: {e}")
                self.training_metrics['SARIMAX'] = {
                    'exog_features': top_features,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
                
        except Exception as e:
            print(f"     Erro no SARIMAX: {e}. Usando ElasticNet como fallback.")
            # Fallback robusto com ElasticNet (regularização L1+L2)
            fallback = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000, random_state=config.RANDOM_STATE)
            fallback.fit(self.X_train, self.y_train)
            self.models['SARIMAX'] = fallback
            self.sarimax_features = self.feature_names  # Usa todas as features
            
            val_preds = fallback.predict(self.X_val)
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
            val_r2 = r2_score(self.y_val, val_preds)
            print(f"     Fallback Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
            self.training_metrics['SARIMAX'] = {
                'fallback': True,
                'val_rmse': val_rmse,
                'val_r2': val_r2
            }

    def train_lstm(self):
        """LSTM proxy com MLPRegressor tunado via RandomizedSearchCV + TimeSeriesSplit."""
        print("  -> Treinando LSTM (MLPRegressor Deep Proxy) com tuning...")
        
        param_dist = {
            'hidden_layer_sizes': [
                (128, 64, 32),
                (256, 128, 64),
                (100, 50, 25),
                (200, 100, 50),
                (64, 32, 16),
                (128, 64),
                (256, 128),
                (512, 256, 128),
            ],
            'activation': ['relu', 'tanh'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['adaptive', 'invscaling'],
            'learning_rate_init': [0.001, 0.005, 0.01],
            'max_iter': [1000, 2000],
            'early_stopping': [True],
            'validation_fraction': [0.15],
            'n_iter_no_change': [20, 30],
            'batch_size': [32, 64, 128]
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        mlp = MLPRegressor(random_state=config.RANDOM_STATE)
        
        random_search = RandomizedSearchCV(
            mlp, param_distributions=param_dist,
            n_iter=30,
            cv=tscv,
            scoring='neg_mean_squared_error',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        self.models['LSTM'] = best_model
        
        # Métricas
        val_preds = best_model.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        val_r2 = r2_score(self.y_val, val_preds)
        
        print(f"     Melhores parâmetros: {random_search.best_params_}")
        print(f"     Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
        
        self.training_metrics['LSTM'] = {
            'best_params': random_search.best_params_,
            'cv_best_score': random_search.best_score_,
            'val_rmse': val_rmse,
            'val_r2': val_r2
        }

    def train_tft(self):
        """TFT proxy com GradientBoosting tunado via RandomizedSearchCV + TimeSeriesSplit."""
        print("  -> Treinando TFT (GradientBoosting Attention Proxy) com tuning...")
        
        param_dist = {
            'n_estimators': [200, 300, 500, 800, 1000],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'loss': ['squared_error', 'huber']
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        gb = GradientBoostingRegressor(random_state=config.RANDOM_STATE)
        
        random_search = RandomizedSearchCV(
            gb, param_distributions=param_dist,
            n_iter=40,
            cv=tscv,
            scoring='neg_mean_squared_error',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        random_search.fit(self.X_train, self.y_train)
        
        best_model = random_search.best_estimator_
        self.models['TFT'] = best_model
        
        # Métricas
        val_preds = best_model.predict(self.X_val)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        val_r2 = r2_score(self.y_val, val_preds)
        
        print(f"     Melhores parâmetros: {random_search.best_params_}")
        print(f"     Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f}")
        
        self.training_metrics['TFT'] = {
            'best_params': random_search.best_params_,
            'cv_best_score': random_search.best_score_,
            'val_rmse': val_rmse,
            'val_r2': val_r2
        }

    def train_all(self):
        """Treina todos os modelos sequencialmente."""
        self.prepare_data()
        
        for model_name in config.MODELS_TO_TRAIN:
            start_time = time.time()
            
            if model_name == 'RandomForest':
                self.train_random_forest()
            elif model_name == 'XGBoost':
                self.train_xgboost()
            elif model_name == 'SARIMAX':
                self.train_sarimax()
            elif model_name == 'LSTM':
                self.train_lstm()
            elif model_name == 'TFT':
                self.train_tft()
            
            elapsed = time.time() - start_time
            if model_name in self.training_metrics:
                self.training_metrics[model_name]['train_time_seconds'] = elapsed
            print(f"     Tempo de treino: {elapsed:.1f}s")
            
        self.save_models()
        return self.models, self.training_metrics

    def save_models(self):
        """Salva modelos treinados e metadados."""
        for name, model in self.models.items():
            filename = f"{name}_{self.dataset_name}_{self.strategy_name}.pkl"
            filepath = os.path.join(config.OUTPUT_DIR, filename)
            
            # Salvar modelo com metadados
            model_data = {
                'model': model,
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'dataset': self.dataset_name,
                'strategy': self.strategy_name,
                'year_col': self.year_col,
                'sarimax_features': self.sarimax_features if name == 'SARIMAX' else None,
                'metrics': self.training_metrics.get(name, {}),
                'y_val': self.y_val,  # Valores reais de validacao
                'X_val': self.X_val   # Features de validacao
            }
            joblib.dump(model_data, filepath)
            print(f"  -> Modelo salvo: {filename}")


def run_training_for_all():
    """Executa treinamento para todos os cenários (datasets × estratégias)."""
    all_training_logs = {}
    
    for dataset in config.DATASETS:
        # Para dataset não agregado, usar apenas estratégia A1
        strategies = ['A1_Direta'] if dataset == 'nao_agregado' else config.STRATEGIES
        
        for strategy in strategies:
            filename = f"{dataset}_{strategy}.csv"
            filepath = os.path.join(config.DATA_DIR, filename)
            
            if os.path.exists(filepath):
                print(f"\n{'='*60}")
                print(f"  TREINANDO: {dataset} - {strategy}")
                print(f"{'='*60}")
                
                df = pd.read_csv(filepath)
                trainer = ModelTrainer(df, dataset, strategy)
                _, metrics = trainer.train_all()
                
                for model_name, data in metrics.items():
                    key = f"{model_name}_{dataset}_{strategy}"
                    all_training_logs[key] = data
            else:
                print(f"\n  AVISO: Ficheiro não encontrado: {filepath}")
                
    # Salvar logs consolidados
    log_path = os.path.join(config.OUTPUT_DIR, 'training_logs.pkl')
    joblib.dump(all_training_logs, log_path)
    print(f"\n{'='*60}")
    print(f"  LOGS DE TREINAMENTO SALVOS: {log_path}")
    print(f"  Total de cenários treinados: {len(all_training_logs)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    run_training_for_all()
