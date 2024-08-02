import pandas as pd
import numpy as nb
from collections import Counter
import matplotlib.pyplot  as mlp
from itertools import combinations
from tabulate import tabulate


# ---------------------------
#Read the percentage file for some visualization
#------------------------------------
top_n = 10
bremen_2011_percent = pd.read_pickle("../../Data/prepared_date/bremen_2011_party_percentages.pkl")
table_2011_party = tabulate(bremen_2011_percent, headers='keys', tablefmt='fancy_grid')
df_2011_sorted = bremen_2011_percent.sort_values(by='percentage %', ascending=False)
top_merged_df_2011 = df_2011_sorted.head(top_n)

bremen_2015_percent = pd.read_pickle("../../Data/prepared_date/bremen_2015_party_percentages.pkl")
table_2015_party = tabulate(bremen_2015_percent, headers='keys', tablefmt='fancy_grid')
df_2015_sorted = bremen_2015_percent.sort_values(by='percentage %', ascending=False)
top_merged_df_2015 = df_2015_sorted.head(top_n)

bremen_2019_percent = pd.read_pickle("../../Data/prepared_date/bremen_2019_party_percentages.pkl")
table_2019_party = tabulate(bremen_2019_percent, headers='keys', tablefmt='fancy_grid')
df_2019_sorted = bremen_2019_percent.sort_values(by='percentage %', ascending=False)
top_merged_df_2019 = df_2019_sorted.head(top_n)

bremen_2023_percent = pd.read_pickle("../../Data/prepared_date/bremen_2023_party_percentages.pkl")
table_2023_party = tabulate(bremen_2023_percent, headers='keys', tablefmt='fancy_grid')
df_2023_sorted = bremen_2023_percent.sort_values(by='percentage %', ascending=False)
top_merged_df_2023 =  df_2023_sorted.head(top_n)


print(table_2011_party)
print(table_2015_party)
print(table_2019_party)
print(table_2023_party)

# ---------------------------------
# Represent the voter percentage of each election cycle in bar charts
# ------------------------------------

# Create a figure and axis for subplots
fig, axs = mlp.subplots(2, 2, figsize=(12, 8))

# Plot each bar chart on its corresponding axis
top_merged_df_2011.plot.bar(x='Kurzform', y='percentage %', rot=0, ax=axs[0, 0])
axs[0, 0].set_title('Bremen 2011')

top_merged_df_2015.plot.bar(x='Kurzform', y='percentage %', rot=0, ax=axs[0, 1])
axs[0, 1].set_title('Bremen 2015')

top_merged_df_2019.plot.bar(x='Kurzform', y='percentage %', rot=0, ax=axs[1, 0])
axs[1, 0].set_title('Bremen 2019')

top_merged_df_2023.plot.bar(x='Kurzform', y='percentage %', rot=0, ax=axs[1, 1])
axs[1, 1].set_title('Bremen 2023')

# Adjust layout to prevent overlapping titles
mlp.tight_layout()

# Show the plot
mlp.show()

import matplotlib.pyplot as plt

# Create a figure and axis for subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Define a function to get the color for each bar
def get_color(row):
    # If the 'Colour' column is NaN, return 'grey', otherwise return the color value
    return 'grey' if pd.isna(row['Colour']) else row['Colour']

# Plot each bar chart on its corresponding axis
for df, ax, title in zip([top_merged_df_2011, top_merged_df_2015, top_merged_df_2019, top_merged_df_2023],
                         axs.flatten(), ['Bremen 2011', 'Bremen 2015', 'Bremen 2019', 'Bremen 2023']):
    df.plot.bar(x='Kurzform', y='percentage %', rot=0, ax=ax, color=df.apply(get_color, axis=1))
    ax.set_title(title)

# Adjust layout to prevent overlapping titles
plt.tight_layout()

# Show the plot
plt.show()


# ---------------------------
# Trying another visual
#------------------

# Combine the datasets
df_2011 = bremen_2011_percent[['Kurzform', 'percentage %', 'Colour']].rename(columns={'percentage %': '2011'})
df_2015 = bremen_2015_percent[['Kurzform', 'percentage %', 'Colour']].rename(columns={'percentage %': '2015'})
df_2019 = bremen_2019_percent[['Kurzform', 'percentage %', 'Colour']].rename(columns={'percentage %': '2019'})
df_2023 = bremen_2023_percent[['Kurzform', 'percentage %', 'Colour']].rename(columns={'percentage %': '2023'})

# Merge the DataFrames on 'Kurzform'
merged_df = df_2011.merge(df_2015, on=['Kurzform', 'Colour'], how='outer')
merged_df = merged_df.merge(df_2019, on=['Kurzform', 'Colour'], how='outer')
merged_df = merged_df.merge(df_2023, on=['Kurzform', 'Colour'], how='outer')

# Fill NaN values with 0 (if a party didn't participate in an election)
merged_df = merged_df.fillna({'2011': 0, '2015': 0, '2019': 0, '2023': 0})

# Handle colors: Fill NaN colors with a default color (e.g., grey for 'Others')
merged_df['Colour'] = merged_df['Colour'].fillna('grey')

# Aggregate parties with grey color into "Others"
others_df = merged_df[merged_df['Colour'] == 'grey']
others_sums = others_df[['2011', '2015', '2019', '2023']].sum()
others_sums['Kurzform'] = 'Others'
others_sums['Colour'] = 'grey'

# Filter out 'Others' from the main DataFrame and add it back separately
filtered_df = merged_df[merged_df['Colour'] != 'grey']
final_df = pd.concat([filtered_df, pd.DataFrame([others_sums])], ignore_index=True)

# Set 'Kurzform' as the index
final_df.set_index('Kurzform', inplace=True)

# Transpose the DataFrame for easier plotting
transposed_df = final_df[['2011', '2015', '2019', '2023']].T

# Plot the line graph
plt.figure(figsize=(12, 8))
for party in transposed_df.columns:
    plt.plot(transposed_df.index, transposed_df[party], marker='o', label=party, color=final_df.loc[party, 'Colour'])

plt.title('Votes of Each Party in Bremen Elections (2011-2023)')
plt.xlabel('Year')
plt.ylabel('Vote Percentage')
plt.legend(title='Party', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()  
