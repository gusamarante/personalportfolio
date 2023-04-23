from data import read_etf, read_ida, tracker_uploader

# BR Equities
bre_names = ['BOVA', 'BCIC', 'BDEF', 'BBSD', 'BREW', 'BRAX', 'BMMT', 'SMAL']
bre_data = read_etf(bre_names)
tracker_uploader(bre_data, 'br equities')

# BR Credit
brc_data = read_ida()
tracker_uploader(brc_data, 'br credit')
