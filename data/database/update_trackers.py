from data import read_etf, tracker_uploader, tracker_feeder

# BR Equities
bre_names = ['BOVA', 'BCIC', 'BDEF', 'BBSD', 'BREW', 'BRAX', 'BMMT', 'SMAL']
bre_data = read_etf(bre_names)
tracker_uploader(bre_data, 'br equities')
