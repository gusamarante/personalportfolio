from data import DROPBOX
from tqdm import tqdm
import pandas as pd

current_year = int(pd.to_datetime('today').year)

def read_ltn_ntnf(start_year=2003, end_year=current_year):
    """
    Reads the raw nominal bond data
    """

    nominal_bonds = pd.DataFrame()

    # LTN
    file_path_ltn = DROPBOX.joinpath('data/LTN')
    for year in tqdm(range(start_year, end_year + 1), 'Reading LTN Data'):
        aux = pd.read_csv(file_path_ltn.joinpath(f'dados_ltn {year}.csv'), sep=';')
        nominal_bonds = pd.concat([nominal_bonds, aux])

    # NTN-F
    file_path_ntnf = DROPBOX.joinpath('data/NTNF')
    for year in tqdm(range(start_year, end_year + 1), 'Reading NTN-F Data'):
        aux = pd.read_csv(file_path_ntnf.joinpath(f'dados_ntnf {year}.csv'), sep=';')
        nominal_bonds = pd.concat([nominal_bonds, aux])

    # Deal with types
    nominal_bonds['reference date'] = pd.to_datetime(nominal_bonds['reference date'])
    nominal_bonds['maturity'] = pd.to_datetime(nominal_bonds['maturity'])
    nominal_bonds = nominal_bonds.drop(['Unnamed: 0', 'index'], axis=1)

    return nominal_bonds


def read_ntnb(start_year=2003, end_year=current_year):
    """
    Reads the raw real bond data
    """

    real_bonds = pd.DataFrame()
    file_path_ltn = DROPBOX.joinpath('data/NTNB')
    for year in tqdm(range(start_year, end_year + 1), 'Reading NTNB Data'):
        aux = pd.read_csv(file_path_ltn.joinpath(f'dados_ntnb {year}.csv'), sep=';')
        real_bonds = pd.concat([real_bonds, aux])

    # Deal with types
    real_bonds['reference date'] = pd.to_datetime(real_bonds['reference date'])
    real_bonds['maturity'] = pd.to_datetime(real_bonds['maturity'])
    real_bonds = real_bonds.drop(['Unnamed: 0', 'index'], axis=1)

    return real_bonds


def read_etf(codes=None):

    file_path_etf = DROPBOX.joinpath('data/ETFs.xlsx')
    df = pd.read_excel(file_path_etf, sheet_name='Prices', skiprows=3, header=0, index_col=0)
    df = df.iloc[2:]
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str[:4]
    df = df.dropna(how='all')

    if codes is not None:
        df = df[codes]

    return df


def read_fip(codes=None):

    file_path_etf = DROPBOX.joinpath('data/FIPs.xlsx')
    df = pd.read_excel(file_path_etf, sheet_name='Trackers', index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str[:4]
    df = df.dropna(how='all')

    if codes is not None:
        df = df[codes]

    return df


def read_fii(codes=None):

    file_path_etf = DROPBOX.joinpath('data/FI-Infra.xlsx')
    df = pd.read_excel(file_path_etf, sheet_name='Trackers', index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str[:4]
    df = df.dropna(how='all')

    if codes is not None:
        df = df[codes]

    return df


def read_ida(codes=None):

    file_path_etf = DROPBOX.joinpath('data/IDA Anbima.xlsx')
    df = pd.read_excel(file_path_etf, sheet_name='Sheet1', skiprows=3, header=0, index_col=0)
    df = df.iloc[2:]
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how='all')

    df = df.rename({'IDADGRAL Index': 'IDA Geral',
                    'IDADDI Index': 'IDA DI',
                    'IDADIPCA Index': 'IDA IPCA'},
                   axis=1)

    if codes is not None:
        df = df[codes]

    return df
