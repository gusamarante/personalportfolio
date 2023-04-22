"""
Creates the empty database with the desired structure.
"""
import sqlite3
from data import grab_connection

conn = grab_connection()
cursor = conn.cursor()

# ===== Create table for the trackers =====
query = "CREATE TABLE trackers (" \
        "'index' TIMESTAMP NOT NULL, " \
        "'variable' TEXT NOT NULL, " \
        "'value' DOUBLE PRECISION NOT NULL, " \
        "PRIMARY KEY ('index', variable)" \
        ");"
try:
        cursor.execute(query)
except sqlite3.OperationalError:
        pass

# ===== Create table for the Focus Survey =====
query = "create table `focus` (" \
        "`indicador` varchar(255) not null, " \
        "`data` DATE not null, " \
        "`datareferencia` DATE not null, " \
        "`basecalculo` INT not null, " \
        "`media` DOUBLE PRECISION not null, " \
        "`mediana` DOUBLE PRECISION not null, " \
        "`desviopadrao` DOUBLE PRECISION not null, " \
        "`minimo` DOUBLE PRECISION not null, " \
        "`maximo` DOUBLE PRECISION not null, " \
        "`numerorespondentes` INT not null," \
        "PRIMARY KEY ('indicador', 'data', 'datareferencia', 'basecalculo')" \
        ");"
try:
        cursor.execute(query)
except sqlite3.OperationalError:
        pass


# ===== Create table for yield curves =====
query = "create table `curves` ( " \
        "`curvename` VARCHAR(255) not null, " \
        "`refdate` DATE not null, " \
        "`daycount` INT not null, " \
        "`yield` DOUBLE PRECISION not null, " \
        "primary key (`curvename`, `refdate`, `daycount`)" \
        ");"
try:
        cursor.execute(query)
except sqlite3.OperationalError:
        pass


# ===== Create table portfolio weights and notionals =====
query = "create table" \
        "`portfolio_composition` (" \
        "`type` VARCHAR(255) not null," \
        "`asset` VARCHAR(255) not null," \
        "`date` DATE not null," \
        "`value` DOUBLE PRECISION not null," \
        "primary key (`type`, `asset`, `date`)" \
        ");"
try:
        cursor.execute(query)
except sqlite3.OperationalError:
        pass


# Close the cursor
cursor.close()