"""
Passo 4: Treinamento de Modelos com Pré-processamento Rigoroso
Implementa pré-processamento específico para cada modelo:
- RandomForest/XGBoost: StandardScaler + detecção de outliers
- SARIMAX: Teste ADF + diferenciação + features estacionárias
- LSTM/TFT: Normalização Min-Max + estrutura temporal
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

import passo4_model_train_config as config

class ModelTrainer:
    def __init__(self, df, dataset_name, strategy_name):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.strategy_name = strategy_name
        self.models = {}
        self.training_metrics = {}
        self.scaler_rf_xgb = StandardScaler()
        self.scaler_lstm_tft = MinMaxScaler()
        self.scaler_sarimax = None  # SARIMAX não usa scaler
        self.feature_names = []
        self.year_col = None
        self.sarimax_features = []
        
    def prepare_data(self):
        """Preparação de dados: separar features, target e identificar coluna temporal."""
        print(f"  Preparando dados para {self.dataset_name} - {self.strategy_name}...")
        
        # Identificar coluna temporal (ano/year)
        self.year_col = 'ano' if 'ano' in self.df.columns else 'year'
        
        # Separar target e features
        if config.TARGET_VAR not in self.df.columns:
            raise ValueError(f"Variável alvo '{config.TARGET_VAR}' não encontrada")
        
        self.y = self.df[config.TARGET_VAR].values
        
        # Features = todas as colunas exceto target e coluna temporal
        feature_cols = [c for c in self.df.columns if c not in [config.TARGET_VAR, self.year_col]]
        self.feature_names = feature_cols
        self.X = self.df[feature_cols].copy()
        
        # Verificar e preencher dados faltantes em FEATURES
        # Estratégia: remover colunas com >50% NaN, depois preencher com forward/backward fill
        if self.X.isnull().sum().sum() > 0:
            total_nan = self.X.isnull().sum().sum()
            print(f"  AVISO: Ainda existem {total_nan} valores faltantes nas features")
            
            # Remover colunas com muitos NaN (>50%)
            nan_threshold = 0.5 * len(self.X)
            cols_to_drop = self.X.columns[self.X.isnull().sum() > nan_threshold]
            if len(cols_to_drop) > 0:
                print(f"     Removendo {len(cols_to_drop)} colunas com >50% NaN")
                self.X = self.X.drop(columns=cols_to_drop)
            
            # Preencher NaN restantes com forward fill, depois backward fill, depois média
            self.X = self.X.fillna(method='ffill').fillna(method='bfill').fillna(self.X.mean())
            
            # Se ainda houver NaN (colunas com todos NaN), preencher com 0
            self.X = self.X.fillna(0)
        
        # Verificar e preencher dados faltantes no TARGET
        y_series = pd.Series(self.y)
        if y_series.isnull().sum() > 0:
            nan_count = y_series.isnull().sum()
            print(f"  AVISO: Existem {nan_count} valores faltantes no target")
            
            # Preencher com forward fill, depois backward fill, depois média
            y_series = y_series.fillna(method='ffill').fillna(method='bfill').fillna(y_series.mean())
            
            # Se ainda houver NaN, preencher com 0
            y_series = y_series.fillna(0)
            self.y = y_series.values
        else:
            self.y = y_series.values
        
        # Remover linhas onde target é NaN ou infinito
        valid_idx = ~(np.isnan(self.y) | np.isinf(self.y))
        if not valid_idx.all():
            print(f"     Removendo {(~valid_idx).sum()} linhas com target inválido")
            self.X = self.X[valid_idx].reset_index(drop=True)
            self.y = self.y[valid_idx]
        
        # Divisão temporal (não aleatória)
        train_end = int(len(self.df) * config.TRAIN_RATIO)
        val_end = train_end + int(len(self.df) * config.VAL_RATIO)
        
        self.X_train_raw = self.X.iloc[:train_end].copy()
        self.y_train = self.y[:train_end]
        self.X_val_raw = self.X.iloc[train_end:val_end].copy()
        self.y_val = self.y[train_end:val_end]
        self.X_test_raw = self.X.iloc[val_end:].copy()
        self.y_test = self.y[val_end:]
        
        # Guardar dados brutos para SARIMAX
        self.X_train_unscaled = self.X_train_raw.copy()
        self.X_val_unscaled = self.X_val_raw.copy()
        self.X_test_unscaled = self.X_test_raw.copy()
        
        # Escalar para RF, XGBoost, LSTM, TFT
        self.X_train_rf_xgb = pd.DataFrame(
            self.scaler_rf_xgb.fit_transform(self.X_train_raw),
            columns=self.feature_names,
            index=self.X_train_raw.index
        )
        self.X_val_rf_xgb = pd.DataFrame(
            self.scaler_rf_xgb.transform(self.X_val_raw),
            columns=self.feature_names,
            index=self.X_val_raw.index
        )
        self.X_test_rf_xgb = pd.DataFrame(
            self.scaler_rf_xgb.transform(self.X_test_raw),
            columns=self.feature_names,
            index=self.X_test_raw.index
        )
        
        # Normalização Min-Max para LSTM/TFT
        self.X_train_lstm_tft = pd.DataFrame(
            self.scaler_lstm_tft.fit_transform(self.X_train_raw),
            columns=self.feature_names,
            index=self.X_train_raw.index
        )
        self.X_val_lstm_tft = pd.DataFrame(
            self.scaler_lstm_tft.transform(self.X_val_raw),
            columns=self.feature_names,
            index=self.X_val_raw.index
        )
        self.X_test_lstm_tft = pd.DataFrame(
            self.scaler_lstm_tft.transform(self.X_test_raw),
            columns=self.feature_names,
            index=self.X_test_raw.index
        )
        
        print(f"  Dados preparados: Treino={len(self.X_train_raw)}, Val={len(self.X_val_raw)}, Teste={len(self.X_test_raw)}")
        print(f"  Features: {len(self.feature_names)} | Target: {config.TARGET_VAR}")

    def _detect_outliers_iqr(self, X, y, multiplier=1.5):
        """Detectar outliers usando IQR e remover."""
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        mask = (y >= lower_bound) & (y <= upper_bound)
        return X[mask], y[mask]

    def train_random_forest(self):
        """Random Forest com GridSearchCV robusto + TimeSeriesSplit."""
        print("  -> Treinando Random Forest...")
        
        # Detectar e remover outliers
        X_train_clean, y_train_clean = self._detect_outliers_iqr(self.X_train_rf_xgb, pd.Series(self.y_train))
        
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
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
        grid_search.fit(X_train_clean, y_train_clean)
        
        best_model = grid_search.best_estimator_
        self.models['RandomForest'] = best_model
        
        # Métricas
        val_preds = best_model.predict(self.X_val_rf_xgb)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        val_r2 = r2_score(self.y_val, val_preds)
        val_mae = mean_absolute_error(self.y_val, val_preds)
        
        print(f"     Melhores parâmetros: {grid_search.best_params_}")
        print(f"     Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f} | Val MAE: {val_mae:.4f}")
        
        self.training_metrics['RandomForest'] = {
            'best_params': grid_search.best_params_,
            'cv_best_score': grid_search.best_score_,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae
        }

    def train_xgboost(self):
        """XGBoost com RandomizedSearchCV robusto + TimeSeriesSplit."""
        print("  -> Treinando XGBoost...")
        
        # Detectar e remover outliers
        X_train_clean, y_train_clean = self._detect_outliers_iqr(self.X_train_rf_xgb, pd.Series(self.y_train))
        
        param_dist = {
            'n_estimators': [100, 200, 300, 500, 800],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9, 11],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.5, 1, 5],
            'reg_alpha': [0, 0.1, 1, 10],
            'reg_lambda': [0, 0.1, 1, 10]
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        xgb_model = xgb.XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            xgb_model, param_distributions=param_dist,
            n_iter=50,
            cv=tscv,
            scoring='neg_mean_squared_error',
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
            refit=True
        )
        random_search.fit(X_train_clean, y_train_clean)
        
        best_model = random_search.best_estimator_
        self.models['XGBoost'] = best_model
        
        # Métricas
        val_preds = best_model.predict(self.X_val_rf_xgb)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        val_r2 = r2_score(self.y_val, val_preds)
        val_mae = mean_absolute_error(self.y_val, val_preds)
        
        print(f"     Melhores parâmetros: {random_search.best_params_}")
        print(f"     Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f} | Val MAE: {val_mae:.4f}")
        
        self.training_metrics['XGBoost'] = {
            'best_params': random_search.best_params_,
            'cv_best_score': random_search.best_score_,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae
        }

    def _check_stationarity(self, series, name=""):
        """Teste ADF para verificar estacionariedade."""
        try:
            result = adfuller(series.dropna(), autolag='AIC')
            p_value = result[1]
            is_stationary = p_value < 0.05
            print(f"     Teste ADF {name}: p-value={p_value:.4f} | Estacionário: {is_stationary}")
            return is_stationary
        except:
            return False

    def _make_stationary(self, series, max_diff=2):
        """Aplicar diferenciação até tornar série estacionária."""
        diff_order = 0
        diff_series = series.copy()
        
        while diff_order < max_diff and not self._check_stationarity(diff_series, f"(diff={diff_order})"):
            diff_series = diff_series.diff().dropna()
            diff_order += 1
        
        return diff_series, diff_order

    def train_sarimax(self):
        """SARIMAX com pré-processamento robusto: teste ADF + diferenciação + features estacionárias."""
        print("  -> Treinando SARIMAX com pré-processamento robusto...")
        try:
            # 1. Verificar estacionariedade da série alvo
            print("     [1/4] Verificando estacionariedade da série alvo...")
            y_train_series = pd.Series(self.y_train)
            is_stationary = self._check_stationarity(y_train_series, "série alvo original")
            
            # 2. Aplicar diferenciação se necessário
            print("     [2/4] Aplicando diferenciação se necessário...")
            if not is_stationary:
                y_train_diff, d_order = self._make_stationary(y_train_series)
                print(f"     Ordem de diferenciação: {d_order}")
            else:
                y_train_diff = y_train_series
                d_order = 0
            
            # 3. Selecionar features exógenas estacionárias
            print("     [3/4] Selecionando features exógenas estacionárias...")
            n_exog = min(5, len(self.feature_names))
            
            # Calcular correlações com target (dados brutos)
            X_train_with_y = self.X_train_unscaled.copy()
            X_train_with_y['__target__'] = self.y_train
            correlations = X_train_with_y.corr()['__target__'].drop('__target__').abs().sort_values(ascending=False)
            
            # Selecionar top features
            top_features = []
            for feat in correlations.index:
                if feat not in ['ano', 'year', self.year_col]:
                    # Verificar se feature é estacionária
                    feat_series = pd.Series(self.X_train_unscaled[feat])
                    if self._check_stationarity(feat_series, f"feature '{feat}'"):
                        top_features.append(feat)
                    elif len(top_features) < n_exog:
                        # Se não estacionária, adicionar mesmo assim (SARIMAX pode lidar)
                        top_features.append(feat)
                
                if len(top_features) >= n_exog:
                    break
            
            self.sarimax_features = top_features
            print(f"     Features exógenas selecionadas: {top_features}")
            
            # 4. Treinar SARIMAX
            print("     [4/4] Treinando SARIMAX...")
            model = SARIMAX(
                endog=self.y_train,
                exog=self.X_train_unscaled[top_features].values if top_features else None,
                order=(1, d_order, 1),  # (p, d, q) - d é ordem de diferenciação detectada
                seasonal_order=(1, 0, 0, 4),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False, maxiter=500)
            
            # Guardar metadados
            fitted_model._exog_features = top_features
            fitted_model._n_exog = len(top_features)
            fitted_model._d_order = d_order
            self.models['SARIMAX'] = fitted_model
            
            # Métricas
            try:
                val_preds = fitted_model.forecast(
                    steps=len(self.y_val),
                    exog=self.X_val_unscaled[top_features].values if top_features else None
                )
                val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
                val_r2 = r2_score(self.y_val, val_preds)
                val_mae = mean_absolute_error(self.y_val, val_preds)
                print(f"     Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f} | Val MAE: {val_mae:.4f}")
                self.training_metrics['SARIMAX'] = {
                    'exog_features': top_features,
                    'd_order': d_order,
                    'val_rmse': val_rmse,
                    'val_r2': val_r2,
                    'val_mae': val_mae,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
            except Exception as e:
                print(f"     Aviso na validação SARIMAX: {e}")
                self.training_metrics['SARIMAX'] = {
                    'exog_features': top_features,
                    'd_order': d_order,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
                
        except Exception as e:
            print(f"     Erro no SARIMAX: {e}. Usando ElasticNet como fallback.")
            # Fallback com ElasticNet
            fallback = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000, random_state=config.RANDOM_STATE)
            fallback.fit(self.X_train_rf_xgb, self.y_train)
            self.models['SARIMAX'] = fallback
            self.sarimax_features = self.feature_names
            
            val_preds = fallback.predict(self.X_val_rf_xgb)
            val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
            val_r2 = r2_score(self.y_val, val_preds)
            val_mae = mean_absolute_error(self.y_val, val_preds)
            print(f"     Fallback Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f} | Val MAE: {val_mae:.4f}")
            self.training_metrics['SARIMAX'] = {
                'fallback': True,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'val_mae': val_mae
            }

    def train_lstm(self):
        """LSTM (MLPRegressor proxy) com normalização Min-Max + tuning."""
        print("  -> Treinando LSTM (MLPRegressor proxy) com normalização Min-Max...")
        
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
        random_search.fit(self.X_train_lstm_tft, self.y_train)
        
        best_model = random_search.best_estimator_
        self.models['LSTM'] = best_model
        
        # Métricas
        val_preds = best_model.predict(self.X_val_lstm_tft)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        val_r2 = r2_score(self.y_val, val_preds)
        val_mae = mean_absolute_error(self.y_val, val_preds)
        
        print(f"     Melhores parâmetros: {random_search.best_params_}")
        print(f"     Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f} | Val MAE: {val_mae:.4f}")
        
        self.training_metrics['LSTM'] = {
            'best_params': random_search.best_params_,
            'cv_best_score': random_search.best_score_,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae
        }

    def train_tft(self):
        """TFT (GradientBoosting proxy) com normalização Min-Max + tuning."""
        print("  -> Treinando TFT (GradientBoosting proxy) com normalização Min-Max...")
        
        param_dist = {
            'n_estimators': [200, 300, 500, 800, 1000],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'validation_fraction': [0.1, 0.15, 0.2],
            'n_iter_no_change': [10, 20, 30]
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
        random_search.fit(self.X_train_lstm_tft, self.y_train)
        
        best_model = random_search.best_estimator_
        self.models['TFT'] = best_model
        
        # Métricas
        val_preds = best_model.predict(self.X_val_lstm_tft)
        val_rmse = np.sqrt(mean_squared_error(self.y_val, val_preds))
        val_r2 = r2_score(self.y_val, val_preds)
        val_mae = mean_absolute_error(self.y_val, val_preds)
        
        print(f"     Melhores parâmetros: {random_search.best_params_}")
        print(f"     Val RMSE: {val_rmse:.4f} | Val R²: {val_r2:.4f} | Val MAE: {val_mae:.4f}")
        
        self.training_metrics['TFT'] = {
            'best_params': random_search.best_params_,
            'cv_best_score': random_search.best_score_,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae
        }

    def train_all_models(self):
        """Treinar todos os 5 modelos."""
        self.prepare_data()
        self.train_random_forest()
        self.train_xgboost()
        self.train_sarimax()
        self.train_lstm()
        self.train_tft()

    def save_models(self, output_dir):
        """Salvar modelos com metadados."""
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            filename = f"{self.dataset_name}_{self.strategy_name}_{model_name}.pkl"
            filepath = os.path.join(output_dir, filename)
            
            # Preparar metadados
            metadata = {
                'model': model,
                'feature_names': self.feature_names,
                'scaler_rf_xgb': self.scaler_rf_xgb,
                'scaler_lstm_tft': self.scaler_lstm_tft,
                'sarimax_features': self.sarimax_features,
                'X_val_unscaled': self.X_val_unscaled,
                'y_val': self.y_val,
                'X_val_rf_xgb': self.X_val_rf_xgb,
                'X_val_lstm_tft': self.X_val_lstm_tft,
                'training_metrics': self.training_metrics.get(model_name, {})
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"  ✓ Modelo salvo: {filename}")


# ============================================================
# FUNÇÃO PRINCIPAL: EXECUTAR TREINAMENTO PARA TODOS OS CENÁRIOS
# ============================================================

def run_training_for_all():
    """
    Executa treinamento completo para todos os cenários:
    - 4 datasets (nao_agregado, inner, left, outer)
    - 3 estratégias (A1_Direta, A2_PCA, A3_Interacao)
    - 5 modelos (RandomForest, XGBoost, SARIMAX, LSTM, TFT)
    Total: 50 modelos
    """
    import os
    import pandas as pd
    
    print("\n" + "="*70)
    print("INICIANDO TREINAMENTO COMPLETO DE MODELOS")
    print("="*70)
    
    # Mapear datasets e suas estratégias
    dataset_strategy_map = {
        'nao_agregado': ['A1_Direta'],
        'inner': ['A1_Direta', 'A2_PCA', 'A3_Interacao'],
        'left': ['A1_Direta', 'A2_PCA', 'A3_Interacao'],
        'outer': ['A1_Direta', 'A2_PCA', 'A3_Interacao']
    }
    
    # Carregar datasets do Passo 3 com padrão correto de nomes
    datasets_to_train = {}
    
    for dataset_name, strategies in dataset_strategy_map.items():
        for strategy_name in strategies:
            # Padrão de nome: {dataset}_{strategy}.csv
            filename = f"{dataset_name}_{strategy_name}.csv"
            dataset_path = os.path.join(config.DATA_DIR, filename)
            
            if not os.path.exists(dataset_path):
                print(f"  ⚠️  Dataset não encontrado: {dataset_path}")
                continue
            
            print(f"\n  Carregando: {filename}")
            df = pd.read_csv(dataset_path)
            datasets_to_train[(dataset_name, strategy_name)] = df
            print(f"     ✓ {df.shape[0]} linhas x {df.shape[1]} colunas")
    
    if not datasets_to_train:
        print("  ❌ Nenhum dataset encontrado em dados_engenharia/")
        print("  Certifique-se de executar o Passo 3 primeiro!")
        return
    
    # Treinar modelos para cada combinação
    total_models = 0
    successful_models = 0
    
    for (dataset_name, strategy_name), df in datasets_to_train.items():
        print(f"\n  {'='*60}")
        print(f"  Treinando: {dataset_name} × {strategy_name}")
        print(f"  {'='*60}")
        
        try:
            # Criar trainer
            trainer = ModelTrainer(df, dataset_name, strategy_name)
            
            # Treinar todos os 5 modelos
            trainer.train_all_models()
            
            # Salvar modelos
            trainer.save_models(config.OUTPUT_DIR)
            
            # Contar modelos treinados
            successful_models += len(trainer.models)
            total_models += 5
            
            print(f"  ✓ {len(trainer.models)} modelos treinados com sucesso")
            
        except Exception as e:
            print(f"  ❌ Erro ao treinar {dataset_name} × {strategy_name}: {e}")
            total_models += 5
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print(f"TREINAMENTO CONCLUÍDO")
    print(f"Modelos treinados com sucesso: {successful_models}/{total_models}")
    print(f"Modelos salvos em: {config.OUTPUT_DIR}")
    print("="*70)
