import util
df = util.read_merged()
clear_df = util.clear_rows(['status_message'], df)
clear_df.to_csv('clear.csv')