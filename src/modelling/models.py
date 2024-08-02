import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import TSNE


#---------------------------
# Load Matrix datasets for the models
#------------------------


df_2011_party_percentages = pd.read_pickle('../../Data/prepared_date/bremen_2011_party_percentages.pkl')

matrix_bre_2011 = pd.read_pickle('../../Data/prepared_date/matrix_bre_2011.pkl')
matrix_bre_2015 = pd.read_pickle('../../Data/prepared_date/matrix_bre_2015.pkl')
matrix_bre_2019 = pd.read_pickle('../../Data/prepared_date/matrix_bre_2019.pkl')
matrix_bre_2023 = pd.read_pickle('../../Data/prepared_date/matrix_bre_2023.pkl')

#---------------------------------
# Party names, labels, total votes, and colors for the visualization
#----------------------------------
bremen_2011_percent = pd.read_pickle("../Data/prepared_date/bremen_2011_party_percentages.pkl")
bre_2011_label_votes = bremen_2011_percent[['Kurzform', 'total_count', 'Colour']]
bremen_2015_percent = pd.read_pickle("../Data/prepared_date/bremen_2015_party_percentages.pkl")
bre_2015_label_votes = bremen_2015_percent[['Kurzform', 'total_count', 'Colour']]
bremen_2019_percent = pd.read_pickle("../Data/prepared_date/bremen_2019_party_percentages.pkl")
bre_2019_label_votes = bremen_2019_percent[['Kurzform', 'total_count', 'Colour']]
bremen_2023_percent = pd.read_pickle("../Data/prepared_date/bremen_2023_party_percentages.pkl")
bre_2023_label_votes = bremen_2023_percent[['Kurzform', 'total_count', 'Colour']]

#------------------------
# Models
#----------------------

def perform_pca_new(matrix_df, labels_df, n_components=2, title='PCA Result', ax=None):
    matrix_array = matrix_df.to_numpy()
    
    covariance_matrix = np.cov(matrix_array, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    topk_indices = sorted_indices[:n_components]
    
    selected_eigenvectors = eigenvectors[:, topk_indices]
    
    # Extract labels and sizes from the labels DataFrame
    labels = labels_df['Kurzform'].tolist()
    sizes = np.sqrt(labels_df['total_count']).tolist() 
    colors = labels_df['Colour'].fillna('grey').tolist()  # Replace NaN with 'grey'
    
    # Plot PCA result with adjusted point sizes and labels
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.scatter(selected_eigenvectors[i, 0], selected_eigenvectors[i, 1], s=sizes[i], color=color)
        ax.annotate(label, (selected_eigenvectors[i, 0], selected_eigenvectors[i, 1]))
    
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.grid(True)

# Define subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

perform_pca_new(matrix_bre_2011, bre_2011_label_votes, title='PCA onAverage Co-voter percentages Bremen 2011 ', ax=axs[0, 0])
perform_pca_new(matrix_bre_2015, bre_2015_label_votes, title='PCA on Average Co-voter percentages Bremen 2015 ', ax=axs[0, 1])
perform_pca_new(matrix_bre_2019, bre_2019_label_votes, title='PCA on Average Co-voter percentages Bremen 2019', ax=axs[1, 0])
perform_pca_new(matrix_bre_2023, bre_2023_label_votes, title='PCA on Average Co-voter percentages Bremen 2023 ', ax=axs[1, 1])

plt.tight_layout()
plt.savefig('PCA on Avg covoter results.png')
plt.show()


# Define subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Perform PCA and plot in subplots
perform_pca_new(matrix_bre_2011_ratio, bre_2011_label_votes, title='PCA on Bremen Voter Ratio 2011', ax=axs[0, 0])
perform_pca_new(matrix_bre_2015_ratio, bre_2015_label_votes, title='PCA on Bremen Voter Ratio 2015', ax=axs[0, 1])
perform_pca_new(matrix_bre_2019_ratio, bre_2019_label_votes, title='PCA on Bremen Voter Ratio 2019', ax=axs[1, 0])
perform_pca_new(matrix_bre_2023_ratio, bre_2023_label_votes, title='PCA on Bremen Voter Ratio 2023', ax=axs[1, 1])

plt.tight_layout()
plt.savefig('PCA on union to int ratio results.png')
plt.show()

#------------------------
# MDS
#-------------

def plot_mds_with_labels_and_sizes(matrix_df, labels_df, title='MDS Result', ax=None):
    matrix_array = matrix_df.to_numpy()
    
    mds = MDS(n_components=2, dissimilarity='precomputed')
    mds_result = mds.fit_transform(matrix_array)
    
    # Extract labels, sizes, and colors from the labels DataFrame
    labels = labels_df['Kurzform'].tolist()
    sizes = np.sqrt(labels_df['total_count']).tolist()
    colors = labels_df['Colour'].fillna('grey').tolist()  # Replace NaN with 'grey'

    # Plot MDS result with adjusted point sizes and colors
    for i, (label, size, color) in enumerate(zip(labels, sizes, colors)):
        ax.scatter(mds_result[i, 0], mds_result[i, 1], s=size, color=color)
        ax.annotate(label, (mds_result[i, 0], mds_result[i, 1]))
    
    # Set plot title and labels
    ax.set_title(title)
    ax.set_xlabel('MDS Dimension 1')
    ax.set_ylabel('MDS Dimension 2')
    ax.grid(True)

# Define subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Perform MDS and plot in subplots
plot_mds_with_labels_and_sizes(matrix_bre_2011, bre_2011_label_votes, title='MDS on Average Co Voter Percentage Bremen 2011', ax=axs[0, 0])
plot_mds_with_labels_and_sizes(matrix_bre_2015, bre_2015_label_votes, title='MDS on Average Co Voter Percentage Bremen 2015', ax=axs[0, 1])
plot_mds_with_labels_and_sizes(matrix_bre_2019, bre_2019_label_votes, title='MDS on Average Co Voter Percentage Bremen 2019', ax=axs[1, 0])
plot_mds_with_labels_and_sizes(matrix_bre_2023, bre_2023_label_votes, title='MDS on Average Co Voter Percentage Bremen 2023', ax=axs[1, 1])

plt.tight_layout()
plt.savefig('MDS on Avg covter percentage.png')
plt.show()

    
    
#------------------------------
# t-SNE 
# ------------------------

def perform_and_plot_tsne(matrix_df, labels_df, labels=None, random_state=0, n_components=2, perplexity=5,
                          learning_rate="auto", title='t-SNE Result', ax=None):
    # Convert dataframe to NumPy array
    matrix_array = matrix_df.to_numpy()
    
    # Perform t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
    tsne_result = tsne.fit_transform(matrix_array)
    
    # Extract labels, sizes, and colors from the labels DataFrame
    labels = labels_df['Kurzform'].tolist()
    sizes = np.sqrt(labels_df['total_count']).tolist()
    colors = labels_df['Colour'].fillna('grey').tolist()  # Replace NaN with 'grey'

    # Plot t-SNE result with adjusted point sizes and colors
    for i, (label, size, color) in enumerate(zip(labels, sizes, colors)):
        ax.scatter(tsne_result[i, 0], tsne_result[i, 1], s=size, color=color)
        ax.annotate(label, (tsne_result[i, 0], tsne_result[i, 1]))
    
    ax.set_title(title)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.grid(True)

# Define subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Perform t-SNE and plot in subplots
perform_and_plot_tsne(matrix_bre_2011, bre_2011_label_votes, title='TSNE on average percent - Bre 2011', ax=axs[0, 0])
perform_and_plot_tsne(matrix_bre_2015, bre_2015_label_votes, title='TSNE on average percent - Bre 2015', ax=axs[0, 1])
perform_and_plot_tsne(matrix_bre_2019, bre_2019_label_votes, title='TSNE on average percent - Bre 2019', ax=axs[1, 0])
perform_and_plot_tsne(matrix_bre_2023, bre_2023_label_votes, title='TSNE on average percent - Bre 2023', ax=axs[1, 1])

plt.tight_layout()
plt.savefig('TSNE on Co Voter percentage.png')
plt.show()


