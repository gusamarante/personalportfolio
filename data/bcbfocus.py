from pandas.tseries.offsets import MonthEnd, YearEnd, QuarterEnd
from requests import get
import pandas as pd


def focus_scrapper(table, max_itens=None):
    """
    Focus Survey API from the BCB
    https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/aplicacao#!/recursos
    """

    bcb_tables = {"IPCA Mensal": "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativaMercadoMensais?$filter=Indicador%20eq%20'IPCA'&$format=json",
                  "IPCA Anual": "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$filter=Indicador%20eq%20'IPCA'&$format=json",
                  "SELIC Anual": "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais?$filter=Indicador%20eq%20'Selic'&$format=json",
                  "SELIC Mensal": ""}  # TODO ADD THIS

    id_vars = ["Indicador", "Data", "DataReferencia", "baseCalculo", "Frequencia"]

    value_vars = ["Media", "Mediana", "DesvioPadrao", "Minimo", "Maximo", "numeroRespondentes"]

    freq_dict = {
        "IPCA Mensal": "Mensal",
        "IPCA Anual": "Anual",
        "SELIC Anual": "Anual"
    }

    assert table in list(bcb_tables.keys()), "Table name not available"

    url = bcb_tables[table]

    if max_itens is not None:
        url = url + f"&$top={max_itens}"

    response = get(url)
    response = response.json()
    response = response["value"]
    df = pd.DataFrame(response)

    df['Frequencia'] = freq_dict[table]

    # Convert type of date variables
    df["Data"] = pd.to_datetime(df["Data"])

    if freq_dict[table] == "Mensal":
        df["DataReferencia"] = pd.to_datetime(df["DataReferencia"]) + MonthEnd(0)

    elif freq_dict[table] == "Anual":
        df["DataReferencia"] = pd.to_datetime(df["DataReferencia"]) + YearEnd(0)

    df = df.drop('IndicadorDetalhe', axis=1)

    return df
