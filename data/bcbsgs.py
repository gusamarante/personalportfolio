import pandas as pd


def sgs(series_id):
    """
    Grabs series from the SGS
    :param series_id: series code on the SGS. (int, str, list of int, list of str or dict)
    :return: pandas DataFrame withe the requested series. If a dict is passed as series ID, the dict values are used
             as column names.
    """
    if type(series_id) is list:  # loop all series codes

        df = pd.DataFrame()
        for cod in series_id:
            single_series = _fetch_single_code(cod)
            df = pd.concat([df, single_series], axis=1)

        df.sort_index(inplace=True)

    elif type(series_id) is dict:

        df = pd.DataFrame()
        for cod in series_id.keys():
            single_series = _fetch_single_code(cod)
            df = pd.concat([df, single_series], axis=1)

        df.columns = series_id.values()

    else:
        df = _fetch_single_code(series_id)

    return df


def _fetch_single_code(series_id):
    """
    Grabs a single series using the API of the SGS. Queries are built using the url
    http://api.bcb.gov.br/dados/serie/bcdata.sgs.{seriesID}/dados?formato=json&dataInicial={initial_date}&dataFinal={end_date}
    The url returns a json file which is read and parsed to a pandas DataFrame.
    """
    url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.' + str(series_id) + '/dados?formato=json'
    df = pd.read_json(url)
    df = df.set_index(pd.to_datetime(df['data'], dayfirst=True)).drop('data', axis=1)
    df.columns = [str(series_id)]
    return df
