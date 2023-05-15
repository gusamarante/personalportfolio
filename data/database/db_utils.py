from pathlib import Path
import sqlite3
import getpass
import os
import pandas as pd

# Creates a constant (DROPBOX) that holds the path to my working directory
username = getpass.getuser()
if os.environ.get('OS') == 'Windows_NT':
    file_path = Path(r'C:\Users\gamarante\Dropbox\Personal Portfolio')
else:  # Assume Mac
    file_path = Path(f'/Users/{username}/Dropbox/Personal Portfolio')  # TODO Update to MAC

DROPBOX = file_path


# Grabs the connection to the database file
def grab_connection():
    """
    grabs the connection to the SQLite file of my personal database
    """
    conn = sqlite3.connect(file_path.joinpath('personal_database.db'))
    return conn


def tracker_delete(names, conn=None):
    """
    Deletes trackers from the databese
    """

    if conn is None:
        conn = grab_connection()

    if isinstance(names, list):
        name_list = "('" + "', '".join(names) + "')"
        query = f"delete from trackers where asset in {name_list}"
    elif isinstance(names, str):
        query = f"delete from trackers where asset = '{names}'"
    else:
        raise ValueError("'names' format is not accepted")

    cursor = conn.cursor()
    cursor.execute(str(query))
    conn.commit()
    cursor.close()


def tracker_uploader(data, pillar_name, conn=None):
    """
    Uploads the tracker data to the database
    """

    # Makes sure that the Index is DateTime
    data.index.name = 'date'
    data.index = pd.to_datetime(list(data.index))

    # If no connection is passed, grabs the default one
    if conn is None:
        conn = grab_connection()

    # Drop the old trackers
    tracker_names = list(data.columns)
    tracker_delete(tracker_names, conn)

    # Put data in the "melted" format
    data = data.melt(ignore_index=False).reset_index()
    data = data.rename({'index': 'date',
                        'variable': 'asset'},
                       axis=1)
    data = data.dropna()
    data['pillar'] = pillar_name

    # upload the new trackers
    data.to_sql('trackers', con=conn, index=False, if_exists='append')


def tracker_feeder(conn=None):
    """
    Read all of the trackers
    """
    # If no connection is passed, grabs the default one
    if conn is None:
        conn = grab_connection()

    query = 'SELECT * FROM trackers'
    df = pd.read_sql(sql=query, con=conn)
    df = df.pivot(index='date', columns=['pillar', 'asset'], values='value')
    df.index = pd.to_datetime(df.index)

    return df


def curve_uploader(data, conn=None, delete_first=False, year=None):
    """
    Uploads curve data to the database
    """

    if delete_first:
        assert year is not None, "A year must be passed in order to delete a curve"

    # If no connection is passed, grabs the default one
    if conn is None:
        conn = grab_connection()

    # Drop the old curve
    if delete_first:
        curve_names = list(data['curvename'].unique())
        if len(curve_names) == 1 and isinstance(curve_names, list):
            curve_names = curve_names[0]
        else:
            raise AssertionError('Only 1 curve can be uploaded at a time')

        curve_delete(names=curve_names, year=year, conn=conn)

    # upload the new trackers
    data.to_sql('curves', con=conn, index=False, if_exists='append')


def curve_delete(names, year, conn=None):
    """
    Deletes curves from the databese
    """

    if conn is None:
        conn = grab_connection()

    query = f"delete from curves " \
            f"where curvename = '{names}' " \
            f"and refdate >= '{year}-01-01' " \
            f"and refdate <= '{year}-12-31';"

    cursor = conn.cursor()
    cursor.execute(str(query))
    conn.commit()
    cursor.close()


def curve_feeder(conn=None):
    """
    Read all of the curves
    """
    # If no connection is passed, grabs the default one
    if conn is None:
        conn = grab_connection()

    query = 'SELECT * FROM curves'
    df = pd.read_sql(sql=query, con=conn)
    df = df.pivot(index='refdate', columns=['curvename', 'daycount'], values='yield')
    df.index = pd.to_datetime(df.index)

    return df


def signal_uploader(data, conn=None):
    """
    Uploads signal data to the database
    """

    signal_name = list(data['signal_name'].unique())

    # If no connection is passed, grabs the default one
    if conn is None:
        conn = grab_connection()

    # Drop the old curve
    signal_delete(signal_name=signal_name, conn=conn)

    # upload the new trackers
    data.to_sql('signals', con=conn, index=False, if_exists='append')


def signal_delete(signal_name, conn):
    signal_name = "('" + "', '".join(signal_name) + "')"

    query = f"delete from signals " \
            f"where signal_name in {signal_name};"

    cursor = conn.cursor()
    cursor.execute(str(query))
    conn.commit()
    cursor.close()


def signal_feeder(conn=None):
    """
    Read all of the signals
    """
    # If no connection is passed, grabs the default one
    if conn is None:
        conn = grab_connection()

    query = 'SELECT * FROM signals'
    df = pd.read_sql(sql=query, con=conn)
    df = df.pivot(index='refdate', columns='signal_name', values='value')
    df.index = pd.to_datetime(df.index)

    return df