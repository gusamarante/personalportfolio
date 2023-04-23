from pathlib import Path
import sqlite3
import getpass
import os


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
