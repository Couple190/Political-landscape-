import pandas as pd


# -----------------------------
# read the party codes
#-----------------------
party_codes = pd.read_excel("../../Data/Parteien-Codes.xlsx")

# --------------------------------
# Function to count the number of uniques numbers in each dataset
# ---------------------------

def count_unique_numbers(df):
  df = df.astype(int)
  
  unique_counts = {}
  
  for index, row in df.iterrows():
      for value in row:
          if value > 0:
              unique_counts[value] = unique_counts.get(value, 0) + 1
          
  unique_counts_df = pd.DataFrame(list(unique_counts.items()), columns=['Number', 'Count'])
  
  return unique_counts_df

# Get the party code from the vote
def get_hundredth(number):
    return (number // 100) *100

#Merged with party code
def party_percentage_dist(df, party_codes, year, district):
    # Merge with the party code
    unique_counts_df = count_unique_numbers(df)
    unique_counts_df['party_id'] = unique_counts_df['Number'].apply(get_hundredth)
    
    party_votes = unique_counts_df.groupby('party_id')['Count'].sum().reset_index()
    party_votes.columns = ['party_id', 'total_count']
    total_votes = party_votes['total_count'].sum()
    party_votes['percentage %'] = ((party_votes['total_count'] / total_votes) * 100).round(2)
    
    party_codes = party_codes[(party_codes['year'] == year) & (party_codes['district'] == district)][['party_id', 'Kurzform', 'Colour']]
    merged_df = pd.merge(party_votes, party_codes, how='inner', on='party_id')
    
    #table_party = tabulate(merged_df, headers='keys', tablefmt='fancy_grid')
    return merged_df

# ----------------------------------------
# Read 2011 Bremen and BremenHaven data
# ----------------------------------------

df_2011 = pd.read_excel("../../Data/Stimmzettel 2011 Wahlbereich Bremen Vertrag.xlsx")
df_2011 = df_2011.iloc[0:]
new_headers = df_2011.iloc[0]
df_2011.columns = new_headers
df_2011 =df_2011[1:]
df_2011.isnull()
df_2011.fillna(0, inplace=True)
Valid_count = df_2011[df_2011['Bemerkungen'] == 0].shape[0]
print(f"There are a total of {df_2011.shape[0]} voters in the 2015 data set of which {Valid_count} successfully cast their votes")
df_2011 = df_2011[df_2011['Bemerkungen'] == 0]
df_2011 = df_2011.drop(columns="Bemerkungen")
df_2011 = df_2011.astype(str) 
df_2011.to_pickle("../../Data/prepared_date/Bremen_2011.pkl")
year = 2011
district = 'Bremen'
Party_percentage_2011 = party_percentage_dist(df_2011, party_codes, year, district)
#print(merged_df_2011)

Party_percentage_2011.to_pickle("../../Data/prepared_date/bremen_2011_party_percentages.pkl")

# -------------------------------
# Read the 2015 Bremen
#------------------------------------

df_2015 = pd.read_excel("../../Data/Stimmzettel 2015 Wahlbereich Bremen Vertrag.xlsx")
df_2015 = df_2015.iloc[0:]
new_headers = df_2015.iloc[0]
df_2015.columns = new_headers
df_2015 =df_2015[1:]
df_2015 = df_2015.reset_index(drop=True)
df_2015.fillna(0, inplace=True)
Valid_count_2015 = df_2015[df_2015['Bemerkungen'] == 0].shape[0]
print(f"There are a total of {df_2015.shape[0]} voters in the 2015 data set of which {Valid_count_2015} successfully cast their votes")
df_2015 = df_2015[df_2015['Bemerkungen'] == 0]
df_2015 = df_2015.drop(columns="Bemerkungen") 
df_2015 = df_2015.astype(str)
df_2015.to_pickle("../../Data/prepared_date/Bremen_2015.pkl")

year = 2015
district = 'Bremen'
Party_percentage_2015 = party_percentage_dist(df_2015, party_codes, year, district)
Party_percentage_2015.to_pickle("../../Data/prepared_date/bremen_2015_party_percentages.pkl")

# -------------------------------
# Read the 2019 Bremen
#------------------------------------

df_2019 = pd.read_excel("../../Data/Stimmzettel 2019 Wahlbereich Bremen Vertrag.xlsx")
df_2019.fillna(0, inplace=True)
Valid_count_2019 = df_2019[df_2019['Grund Ungültigkeit'] == 0].shape[0]
print(f"There are a total of {df_2019.shape[0]} voters in the 2019 data set of which {Valid_count_2019} successfully cast their votes")
df_2019 = df_2019[df_2019['Grund Ungültigkeit'] == 0]
df_2019 = df_2019.drop(columns="Grund Ungültigkeit") 
df_2019.rename(columns={'Stimme1': 'Stimme 1', 
                                  'Stimme2':'Stimme 2', 
                                  'Stimme3':'Stimme 3', 
                                  'Stimme4':'Stimme 4', 
                                  'Stimme5':'Stimme 5'}, inplace=True)
df_2019 = df_2019.astype(int).astype(str)
df_2019.to_pickle("../../Data/prepared_date/Bremen_2019.pkl")

year = 2019
district = 'Bremen'
Party_percentage_2019 = party_percentage_dist(df_2019, party_codes, year, district)
Party_percentage_2019.to_pickle("../../Data/prepared_date/bremen_2019_party_percentages.pkl")

# -------------------------------
# Read the 2023 Bremen
#------------------------------------

df_2023 = pd.read_excel("../../Data/Stimmzettel 2023 Wahlbereich Bremen Vertrag.xlsx")
df_2023 = df_2023.iloc[0:]
new_headers = df_2023.iloc[0]
df_2023.columns = new_headers
df_2023 =df_2023[1:]
df_2023 = df_2023.reset_index(drop=True)
df_2023.drop(df_2023[df_2023['Gültigkeit'] == 'Ungültig (per Beschluss)'].index, inplace = True)
df_2023.drop(df_2023[df_2023['Gültigkeit'] == 'Ungültig'].index, inplace = True)
df_2023 = df_2023.drop(columns="Gültigkeit")
df_2023 = df_2023.drop(columns="Bemerkung")
df_2023.fillna(0, inplace=True)
df_2023 = df_2023.astype(str)
df_2023.to_pickle("../../Data/prepared_date/Bremen_2023.pkl")

year = 2023
district = 'Bremen'
Party_percentage_2023 = party_percentage_dist(df_2023, party_codes, year, district)
Party_percentage_2023.to_pickle("../../Data/prepared_date/bremen_2023_party_percentages.pkl")
