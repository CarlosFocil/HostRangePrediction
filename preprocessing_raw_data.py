import pandas as pd

# Load raw data from excel file
aerobic_data = pd.read_excel('raw_data/seif_support_catabolic-data.xlsx', sheet_name=2, header=2, index_col=0)
aerobic_data.columns = aerobic_data.columns.str.replace(' ','_').str.lower()
aerobic_data = aerobic_data.add_suffix('(O2+)')

good_columns = aerobic_data.sum()[aerobic_data.sum().round(5) > 0.01].index.to_list()
clean_aerobic = aerobic_data[good_columns]


anaerobic_data = pd.read_excel('raw_data/seif_support_catabolic-data.xlsx', sheet_name=3, header=2, index_col=0)
anaerobic_data.columns = anaerobic_data.columns.str.replace(' ','_').str.lower()
anaerobic_data = anaerobic_data.add_suffix('(O2-)')

good_columns = anaerobic_data.sum()[anaerobic_data.sum().round(5) > 0.01].index.to_list()
clean_anaerobic = anaerobic_data[good_columns]

clean_aerobic = clean_aerobic.reset_index(names='strain_id')
clean_anaerobic = clean_anaerobic.reset_index(names='strain_id')

joined_df = pd.merge(clean_aerobic, clean_anaerobic, on ='strain_id')

metadata = pd.read_excel('raw_data/seif_metadata.xlsx', sheet_name=0, header=1)
metadata.columns = metadata.columns.str.lower().str.replace(' ','_')
columns = ['genome_accession', 'host_range']
df2 = metadata[columns]
df2 = df2.rename(columns={'genome_accession':'strain_id'})

full_df = pd.merge(joined_df, df2, on='strain_id')
full_df = full_df.loc[full_df.host_range.dropna().index].reset_index(drop=True)

full_df.to_csv('pre-processed_strain_profiles.csv')