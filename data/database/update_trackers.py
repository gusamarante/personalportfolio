from data import read_etf, read_ida, tracker_uploader, read_managers


# BR Equities
bre_names = ['BOVA', 'BCIC', 'BDEF', 'BBSD', 'BREW', 'BRAX', 'BMMT', 'SMAL']
bre_data = read_etf(bre_names)
tracker_uploader(bre_data, 'br equities')


# BR Credit
brc_data = read_ida()
tracker_uploader(brc_data, 'br credit')


# BR Managers
brm_data = read_managers()
tracker_uploader(brc_data, 'br managers')


# Global Equities
gle_names = ['XINA', 'ASIA', 'EURP', 'ACWI', 'NASD', 'IVVB']
gle_data = read_etf(gle_names)
tracker_uploader(gle_data, 'global equities')
