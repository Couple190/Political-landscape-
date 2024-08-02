import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot  as mlp
from itertools import combinations
from tabulate import tabulate
from sklearn.manifold import TSNE


# -----------------------------
# read the party codes
#-----------------------
party_codes = pd.read_excel("../../Data/Parteien-Codes.xlsx")
party_codes_2011 = party_codes[(party_codes['year'] == 2011) & (party_codes['district'] == 
                    'Bremen')][['party_id', 'Kurzform', 'Colour']]
party_codes_2015 = party_codes[(party_codes['year'] == 2015) & (party_codes['district'] == 
                    'Bremen')][['party_id', 'Kurzform',  'Colour']]
party_codes_2019 = party_codes[(party_codes['year'] == 2019) & (party_codes['district'] == 
                    'Bremen')][['party_id', 'Kurzform',  'Colour']]
party_codes_2023 = party_codes[(party_codes['year'] == 2023) & (party_codes['district'] == 
                    'Bremen')][['party_id', 'Kurzform',  'Colour']]
#--------------------
# Load datasets and get the numbers of voters for each party
#------------------------

df_2011 = pd.read_pickle("../../Data/prepared_date/Bremen_2011.pkl")
df_2015 = pd.read_pickle("../../Data/prepared_date/Bremen_2015.pkl")
df_2019 = pd.read_pickle("../../Data/prepared_date/Bremen_2019.pkl")
df_2023 = pd.read_pickle("../../Data/prepared_date/Bremen_2023.pkl")


def convert_to_hundredth(df):
    df = df.applymap(int)
    
    df = df.applymap(lambda x: (x // 100) * 100)
    
    return df

df_bre_2011 = convert_to_hundredth(df_2011)
df_bre_2015 = convert_to_hundredth(df_2015)
df_bre_2019 = convert_to_hundredth(df_2019)
df_bre_2023 = convert_to_hundredth(df_2023)

############ -----------------------------
#Convert the party codes to names
#---------------------------

def replace_codes_with_names(df_data, df_codes):
    merged_columns = []

    for code_column in ['Stimme 1', 'Stimme 2', 'Stimme 3', 'Stimme 4', 'Stimme 5']:
        merged_df = pd.merge(df_data, df_codes, how='left', left_on=code_column, right_on='party_id')
        
        merged_df.drop(columns=[col for col in merged_df.columns if col != 'Kurzform'], inplace=True)
        
        merged_df.rename(columns={'Kurzform': code_column}, inplace=True)
        
        merged_columns.append(merged_df)

    final_df = pd.concat(merged_columns, axis=1)
    
    return final_df

df_bre_2011_prty = replace_codes_with_names(df_bre_2011, party_codes_2011)
df_bre_2015_prty = replace_codes_with_names(df_bre_2015, party_codes_2015)
df_bre_2019_prty = replace_codes_with_names(df_bre_2019, party_codes_2019)
df_bre_2023_prty = replace_codes_with_names(df_bre_2023, party_codes_2023)

#-------------------------
# Create a matrix of votes each voter gave a party
#------------------------

def count_unique_values(df):
    # Get unique values from the original dataset
    unique_values = df.stack().unique()

    # Create a new DataFrame with unique values as columns
    new_df = pd.DataFrame(columns=unique_values)
    
    #new_df = pd.concat([df, new_df], axis=1)


    return new_df

df_bre_prty_votr = count_unique_values(df_bre_2011_prty)

def count_occurrences_and_create_new_df(df):
    df = df.dropna(axis=0)
    # Get unique values from the original dataset
    unique_values = df.stack().unique()

    # Initialize a dictionary to store counts of unique values
    counts_dict = {value: [0] * len(df) for value in unique_values}

    # Loop through each row in the original DataFrame
    for index, row in df.iterrows():
        # Count the occurrences of each unique value in the row
        counts = row.value_counts()

        # Update the counts dictionary with the row counts
        for value, count in counts.items():
            # Check if the value exists in the dictionary
            if value in counts_dict:
                # Check if the index is within the range of the list
                if index < len(counts_dict[value]):
                    counts_dict[value][index] = count

    # Create a new DataFrame from the counts dictionary
    new_df = pd.DataFrame(counts_dict)

    return new_df

df_bre_2011_prty_votr = count_occurrences_and_create_new_df(df_bre_2011_prty)
df_bre_2015_prty_votr = count_occurrences_and_create_new_df(df_bre_2015_prty)
df_bre_2019_prty_votr = count_occurrences_and_create_new_df(df_bre_2019_prty)
df_bre_2023_prty_votr = count_occurrences_and_create_new_df(df_bre_2023_prty)
df_bre_2011_prty_votr.to_pickle("../../Data/prepared_date/bremen_2011_matrix_data.pkl")
df_bre_2015_prty_votr.to_pickle("../../Data/prepared_date/bremen_2015_matrix_data.pkl")
df_bre_2019_prty_votr.to_pickle("../../Data/prepared_date/bremen_2019_matrix_data.pkl")
df_bre_2023_prty_votr.to_pickle("../../Data/prepared_date/bremen_2023_matrix_data.pkl")

def count_total_voters(df):
    df = df.dropna()

    unique_counts = {}

    for _, row in df.iterrows():
        unique_values_in_row = set(row)

        for value in unique_values_in_row:
            unique_counts[value] = unique_counts.get(value, 0) + 1

    # Convert the dictionary to a DataFrame
    unique_counts_df = pd.DataFrame(list(unique_counts.items()), columns=['Party_id', 'Total_voters'])
    #df_party = df_party_pairs.merge(df_total_voters, how='left', left_on='Party_Code_A', right_on='Party_id')

    return unique_counts_df

df_2011_total_voters = count_total_voters(df_bre_2011_prty)
df_2015_total_voters = count_total_voters(df_bre_2015_prty)
df_2019_total_voters = count_total_voters(df_bre_2019_prty)
df_2023_total_voters = count_total_voters(df_bre_2023_prty)


#---------------------------------
# Generate Uniqure pairs of parties that voters voted for
# ---------------------------

def unique_pairs_with_count(df):
    df = df.applymap(int)
    # Initialize a Counter to store the counts of unique pairs
    pair_counter = Counter()

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        # Convert row values to a list and change the last two digits of each value to 00
        row_values = [(x // 100) * 100 for x in row if x!= 0]

        # Generate all unique pairs within the row using combinations
        unique_pairs = set(combinations(row_values, 2))

        # Update the pair_counter with the counts of these unique pairs
        pair_counter.update(unique_pairs)

    # Convert the Counter to a DataFrame
    pair_counts_df = pd.DataFrame(pair_counter.items(), columns=['Pair', 'Count'])

    pair_counts_df[['Party_Code_A', 'Party_Code_B']] = pd.DataFrame(pair_counts_df['Pair'].tolist())
    pair_counts_df = pair_counts_df[['Party_Code_A', 'Party_Code_B', 'Count']]
    
    return pair_counts_df

df_2011_pair = unique_pairs_with_count(df_2011)
df_2015_pair = unique_pairs_with_count(df_2015)
df_2019_pair = unique_pairs_with_count(df_2019)
df_2023_pair = unique_pairs_with_count(df_2023)

#------------------------
# Replace party codes with party names in the covoter dataframe 
#-------------------------

def merge_and_clean(df, party_codes):
    #df_merged= df.merge(party_votes, how='left', left_on ='Party_Code_A', right_on= ['Party_id'])
    # Merge with party codes for party A
    df_merged = df.merge(party_codes, how='left', left_on='Party_Code_A', right_on='party_id')
    df_merged.drop(columns=['party_id'], inplace=True)
    df_merged.rename(columns={'Kurzform': 'party_A'}, inplace=True)
    
    # Merge with party codes for party B
    df_merged = df_merged.merge(party_codes, how='left', left_on='Party_Code_B', right_on='party_id')
    df_merged.drop(columns=['party_id'], inplace=True)
    df_merged.rename(columns={'Kurzform': 'party_B'}, inplace=True)
    
    return df_merged

# Call the function with your dataframes
df_2011_party_pairs = merge_and_clean(df_2011_pair, party_codes_2011)
df_2015_party_pairs = merge_and_clean(df_2015_pair, party_codes_2015)
df_2019_party_pairs = merge_and_clean(df_2019_pair, party_codes_2019)
df_2023_party_pairs = merge_and_clean(df_2023_pair, party_codes_2023)

# --------------------------------
# Get the avaerage percentage of co voters to compute the matrix
# --------------------------------

def compute_party_voting_percentages(df_party_pairs, df_total_voters):
    # Merge with total voters for party A
    df_party = df_party_pairs.merge(df_total_voters, how='left', left_on='party_A', right_on='Party_id')
    
    # Calculate percentage of party A voted for party B
    df_party['Percentage_A_voted_B'] = (df_party['Count'] / df_party['Total_voters']) * 100
    
    # Drop unnecessary columns
    df_party.drop(columns=['Total_voters'], inplace=True)
    
    # Merge with total voters for party B
    df_party = df_party.merge(df_total_voters, how='left', left_on='party_B', right_on='Party_id', suffixes=('_A', '_B'))
    
    # Calculate percentage of party B voted for party A
    df_party['Percentage_B_voted_A'] = (df_party['Count'] / df_party['Total_voters']) * 100
    
    # Drop unnecessary columns
    df_party.drop(columns=['Total_voters'], inplace=True)
    
    # Select the desired columns
    df_party_result = df_party[['party_A', 'party_B', 'Count', 'Percentage_A_voted_B', 'Percentage_B_voted_A']]
    
    # Compute the average percentage and round to one decimal place
    df_party_result['Average_percentage'] = ((df_party_result['Percentage_A_voted_B'] + df_party_result['Percentage_B_voted_A']) / 200).round(1)
    
    return df_party_result


df_2011_party_percentages = compute_party_voting_percentages(df_2011_party_pairs,df_2011_total_voters)
df_2015_party_percentages = compute_party_voting_percentages(df_2015_party_pairs,df_2015_total_voters)
df_2019_party_percentages = compute_party_voting_percentages(df_2019_party_pairs,df_2019_total_voters)
df_2023_party_percentages = compute_party_voting_percentages(df_2023_party_pairs,df_2023_total_voters)

#---------------------------------
# Party names, labels. total votes, and colors for the visualization
#----------------------------------
bremen_2011_percent = pd.read_pickle("../../Data/prepared_date/bremen_2011_party_percentages.pkl")
#bre_2011_label_votes = bremen_2011_percent.merge(party_codes_2011, how= 'left', on='Kurzform')
bre_2011_label_votes = bremen_2011_percent[['Kurzform', 'total_count', 'Colour']]
bremen_2015_percent = pd.read_pickle("../../Data/prepared_date/bremen_2015_party_percentages.pkl")
#bre_2015_label_votes = bremen_2015_percent.merge(party_codes_2015, how= 'left', on='Kurzform')
bre_2015_label_votes = bremen_2015_percent[['Kurzform', 'total_count', 'Colour']]
bremen_2019_percent = pd.read_pickle("../../Data/prepared_date/bremen_2019_party_percentages.pkl")
#bre_2019_label_votes = bremen_2019_percent.merge(party_codes_2019, how= 'left', on='Kurzform')
bre_2019_label_votes = bremen_2019_percent[['Kurzform', 'total_count', 'Colour']]
bremen_2023_percent = pd.read_pickle("../../Data/prepared_date/bremen_2023_party_percentages.pkl")
#bre_2023_label_votes = bremen_2023_percent.merge(party_codes_2023, how= 'left', on='Kurzform')
bre_2023_label_votes = bremen_2023_percent[['Kurzform', 'total_count', 'Colour']]
#--------------------------------
# Compute covoter percentage matrix
#----------------------------------

def create_percentage_matrix(df):
    # Get unique values of party_A and party_B
    unique_party_A = df['party_A'].unique()
    unique_party_B = df['party_B'].unique()
    
    # Initialize an empty DataFrame with dimensions len(unique_party_A) x len(unique_party_B)
    matrix = pd.DataFrame(index=unique_party_A, columns=unique_party_A)
    
    # Iterate over each unique pair of parties and fill in the matrix with the average percentage
    for party_A in unique_party_A:
        for party_B in unique_party_A:
            # Find the corresponding row in the DataFrame
            subset_df = df[(df['party_A'] == party_A) & (df['party_B'] == party_B)]
            # Check if there are any rows in the subset DataFrame
            if not subset_df.empty:
                # Get the average percentage for the pair
                average_percentage = subset_df['Average_percentage'].values[0]  # Assuming there's only one value
                # Fill in the matrix
                matrix.loc[party_A, party_B] = average_percentage
    matrix.fillna(0, inplace=True)
    # Mirror the lower triangle to the upper triangle
    for i in range(len(matrix)):
        for j in range(i):
            matrix.iloc[i, j] = matrix.iloc[j, i]
    
    return matrix


# Call the function with your DataFrame
matrix_bre_2011 = create_percentage_matrix(df_2011_party_percentages)
matrix_bre_2015 = create_percentage_matrix(df_2015_party_percentages)
matrix_bre_2019 = create_percentage_matrix(df_2019_party_percentages)
matrix_bre_2023 = create_percentage_matrix(df_2023_party_percentages)
matrix_bre_2011.to_pickle("../../Data/prepared_date/matrix_bre_2011.pkl")
matrix_bre_2015.to_pickle("../../Data/prepared_date/matrix_bre_2015.pkl")
matrix_bre_2019.to_pickle("../../Data/prepared_date/matrix_bre_2019.pkl")
matrix_bre_2023.to_pickle("../../Data/prepared_date/matrix_bre_2023.pkl")

print(f"This gives a Upper truangular matrix or Right trainagular, so I mirrowed it to get a full matrix")



#--------------------------------------
# get function to create a matrix for the ration of the common voters to all voter
#----------------------------------------- 

def count_votes_for_pairs(df):
    df = df.applymap(int)
    pair_counter = Counter()
    df['party_a_intersection_party_b'] = 0
    df['party_a_union_party_b'] = 0 
    for index, row in df.iterrows():
        row_values = [(x // 100) * 100 for x in row if x!= 0]   
        unique_pairs = set(combinations(row_values, 2)) 
        pair_counter.update(unique_pairs)   
    pair_counts_df = pd.DataFrame(pair_counter.items(), columns=['Pair', 'Count'])  
    pair_counts_df[['Party_Code_A', 'Party_Code_B']] = pd.DataFrame(pair_counts_df['Pair'].tolist())
    #pair_counts_df = pair_counts_df[['Party_Code_A', 'Party_Code_B', 'Count']]
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        party_a = row['Party_Code_A']
        party_b = row['Party_Code_B']
        
        # Count votes for both parties and votes for either party
        votes_for_both = df[(df['Party_Code_A'] == party_a) & (df['Party_Code_B'] == party_b)]['Count'].sum()
        votes_for_either = df[(df['Party_Code_A'] == party_a) | (df['Party_Code_B'] == party_b)]['Count'].sum()
        ratio_f = (votes_for_both / votes_for_either).round(2)
        # Update the new columns with counts
        df.loc[index, 'party_a_intersection_party_b'] = votes_for_both
        df.loc[index, 'party_a_union_party_b'] = votes_for_either
        df.loc[index, 'party_votes_ratio'] = ratio_f
       
    return df


df_2011_pair_union = count_votes_for_pairs(df_2011_pair)
df_2015_pair_union = count_votes_for_pairs(df_2015_pair)
df_2019_pair_union = count_votes_for_pairs(df_2019_pair)
df_2023_pair_union = count_votes_for_pairs(df_2023_pair)


df_2011_party_pairs_ratio = merge_and_clean(df_2011_pair_union, party_codes_2011)
df_2015_party_pairs_ratio = merge_and_clean(df_2015_pair_union, party_codes_2015)
df_2019_party_pairs_ratio = merge_and_clean(df_2019_pair_union, party_codes_2019)
df_2023_party_pairs_ratio = merge_and_clean(df_2023_pair_union, party_codes_2023)

def create_ratio_matrix(df):
    # Get unique values of party_A and party_B
    unique_party_A = df['party_A'].unique()
    unique_party_B = df['party_B'].unique()

    matrix = pd.DataFrame(index=unique_party_A, columns=unique_party_B)
    
    # Iterate over each unique pair of parties and fill in the matrix with the average percentage
    for party_A in unique_party_A:
        for party_B in unique_party_B:
            subset_df = df[(df['party_A'] == party_A) & (df['party_B'] == party_B)]
            if not subset_df.empty:
                voter_ratio = subset_df['party_votes_ratio'].values[0]   
                matrix.loc[party_A, party_B] = voter_ratio
    
    matrix.fillna(0, inplace=True)

    # Mirror the lower triangle to the upper triangle
    for i in range(len(matrix)):
        for j in range(i):
            matrix.iloc[i, j] = matrix.iloc[j, i]
    
    return matrix

matrix_bre_2011_ratio = create_ratio_matrix(df_2011_party_pairs_ratio)
matrix_bre_2015_ratio = create_ratio_matrix(df_2015_party_pairs_ratio)
matrix_bre_2019_ratio = create_ratio_matrix(df_2019_party_pairs_ratio)
matrix_bre_2023_ratio = create_ratio_matrix(df_2023_party_pairs_ratio) 
matrix_bre_2011_ratio.to_pickle("../../Data/prepared_date/matrix_bre_2011_ratio.pkl")
matrix_bre_2015_ratio.to_pickle("../../Data/prepared_date/matrix_bre_2015_ratio.pkl")
matrix_bre_2019_ratio.to_pickle("../../Data/prepared_date/matrix_bre_2019_ratio.pkl")
matrix_bre_2023_ratio.to_pickle("../../Data/prepared_date/matrix_bre_2023_ratio.pkl")
