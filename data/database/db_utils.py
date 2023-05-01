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


def curve_uploader(data, conn=None, delete_first=True): # TODO option to delet first
    """
    Uploads curve data to the database
    """

    # If no connection is passed, grabs the default one
    if conn is None:
        conn = grab_connection()

    # Drop the old curve
    if delete_first:
        curve_names = list(data['curvename'].unique())
        curve_delete(curve_names, conn)

    # upload the new trackers
    data.to_sql('curves', con=conn, index=False, if_exists='append')


def curve_delete(names, conn=None):
    """
    Deletes curves from the databese
    """

    if conn is None:
        conn = grab_connection()

    if isinstance(names, list):
        name_list = "('" + "', '".join(names) + "')"
        query = f"delete from curves where curvename in {name_list}"
    elif isinstance(names, str):
        query = f"delete from curves where curvename = '{names}'"
    else:
        raise ValueError("'names' format is not accepted")

    cursor = conn.cursor()
    cursor.execute(str(query))
    conn.commit()
    cursor.close()
