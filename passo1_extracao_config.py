"""Configurações do projeto."""

BASE_URL = "https://api.worldbank.org/v2"
DATA_DIR = "data/raw"

INDICADORES = {
    "NY.GDP.PCAP.PP.KD": "pib_per_capita_ppc",
    "NE.GDI.FTOT.ZS": "formacao_bruta_capital_fixo_percent_pib",
    "SE.SEC.ENRR": "matricula_ensino_secundario_percent",
    "NE.TRD.GNFS.ZS": "comercio_percent_pib",
    "BX.KLT.DINV.WD.GD.ZS": "investimento_estrangeiro_direto_percent_pib",
    "SP.POP.TOTL": "populacao_total",
    "SL.IND.EMPL.ZS": "emprego_industria_percent_emprego_total",
    "NV.IND.TOTL.ZS": "valor_agregado_industrial_percent_pib",
}

DATA_INICIO = 1996
DATA_FIM = 2023
