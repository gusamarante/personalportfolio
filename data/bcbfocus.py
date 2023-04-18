from pandas.tseries.offsets import MonthEnd, YearEnd, QuarterEnd
from requests import get
import pandas as pd


class FocusScrapper(object):
    """
    New API from the BCB
    https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/aplicacao#!/recursos
    """

    bcb_tables = {"mensal": "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativaMercadoMensais?$format=json",
                  "selic": "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoSelic?$format=json",
                  "trimestral": "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoTrimestrais?$format=json",
                  "anual": "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$format=json"}

    rename_dict = {"Indicador": "index",
                   "Data": "time_stamp",
                   "DataReferencia": "prediction_scope",
                   "baseCalculo": "survey_type"}

    frequencies = {"mensal": "mensal",
                   "trimestral": "trimestral",
                   "anual": "anual"}

    id_vars = ["time_stamp", "index", "prediction_scope", "survey_type"]

    value_vars = ["Media", "Mediana", "DesvioPadrao", "Minimo", "Maximo", "numeroRespondentes"]

    def __init__(self, max_itens=10000, bcb_table="mensal"):

        assert bcb_table in list(self.bcb_tables.keys())

        url = self.bcb_tables[bcb_table]
        if max_itens is not None:
            url = url + f"&$top={max_itens}"

        response = get(url)
        response = response.json()
        response = response["value"]
        self.df = pd.DataFrame(response)
        self.df = self.df.rename(self.rename_dict, axis=1)
        self.df = self.df.melt(id_vars=self.id_vars, value_vars=self.value_vars, var_name="metric")
        self.df["frequency"] = self.frequencies[bcb_table]

        if bcb_table == "mensal":
            self.df["prediction_scope"] = pd.to_datetime(self.df["prediction_scope"]) + MonthEnd(0)

        elif bcb_table == "trimestral":
            self.df["prediction_scope"] = pd.to_datetime(self.df["prediction_scope"]) + QuarterEnd(0)

        elif bcb_table == "anual":
            self.df["prediction_scope"] = pd.to_datetime(self.df["prediction_scope"]) + YearEnd(0)

        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates(subset=["time_stamp", "index", "prediction_scope", "survey_type", "metric", "frequency"])

        self.df["index"] = self.df["index"].str.lower()
        self.df["metric"] = self.df["metric"].str.lower()
        self.df["frequency"] = self.df["frequency"].str.lower()

        self.df["time_stamp"] = pd.to_datetime(self.df["time_stamp"])
        self.df["prediction_scope"] = pd.to_datetime(self.df["prediction_scope"])

        # Upload to DB
        # TODO - How do I save this?
        a = 1
