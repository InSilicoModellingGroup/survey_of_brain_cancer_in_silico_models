# !pip3 install -U kaleido
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import umap
import plotly.graph_objects as go

custom_colors = [
    '#636efa', '#ef553b', '#00cc96', '#ffa15a','#ab63fa', 
    '#1bd3f3', '#ff6c96', '#cbeea5', '#fe9eff'
]

# ----------------------------------------------------------------------
def compute_cluster_feature_importances(X, y):
    cluster_feature_importances = {}

    # Train a classifier for each cluster
    for cluster in np.unique(y):
        binary_labels = (y == cluster).astype(int)  # Create binary labels (1 for current cluster, 0 for all others)

        X_train, X_test, y_train, y_test = train_test_split(X, binary_labels, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)  # Train a Random Forest Classifier
        clf.fit(X_train, y_train)

        feature_importances = clf.feature_importances_  # Retrieve feature importance scores

        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})  # Store the results in a DataFrame for better readability
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        cluster_feature_importances[cluster] = importance_df  # Store the DataFrame in the dictionary with the cluster as the key

    return cluster_feature_importances

# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
import plotly.io as pio

def plot_umap_clusters(data_scaled, data_original, type, output_dir, n_neighbors=15, min_dist=0.01, n_components=3):
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the UMAP reducer
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    
    # Fit and transform the data
    data_umap = reducer.fit_transform(data_scaled)
    
    # Create a 3D figure object
    fig = go.Figure()

    # Determine cluster type
    clustertype = f'{type}_Cluster'
    
    # Identify unique clusters and sort them
    clusters = sorted(data_original[clustertype].unique())
    
    # Loop through each sorted cluster and add separate traces
    for i, cluster in enumerate(clusters):
        cluster_data = data_original[data_original[clustertype] == cluster]
        indices = cluster_data.index
        cluster_umap = data_umap[indices]
        
        # Pick color from custom_colors, using modulo to prevent index errors 
        # if there are more clusters than colors
        trace_color = custom_colors[i % len(custom_colors)]
        
        # Add the trace
        fig.add_trace(go.Scatter3d(
            x=cluster_umap[:, 0],
            y=cluster_umap[:, 1],
            z=cluster_umap[:, 2],
            mode='markers',
            marker=dict(
                size=(cluster_data['Year'] - cluster_data['Year'].min()) / 
                     (cluster_data['Year'].max() - cluster_data['Year'].min()) * 20,
                color=trace_color,  # Set the custom color here
                opacity=0.7
            ),
            name=f'Cluster {cluster}',  # Name each trace with the cluster number
            text=[f'<b>Cluster {cluster}</b> <br>{title}<br>{authors}<br>{year}' for year, title, authors in zip(cluster_data['Year'], cluster_data['Title'], cluster_data['Authors'])],
            hovertemplate='%{text}<br><extra></extra>'  # Display custom hover text
        ))
    
    # Update the layout of the figure
    fig.update_layout(
        title=f"UMAP Projection using {len(clusters)} {type} Clusters - Size by Year",
        scene=dict(
            xaxis=dict(
                title='UMAP1',
                showbackground=True  # Remove background plane for x-axis
            ),
            yaxis=dict(
                title='UMAP2',
                showbackground=True  # Remove background plane for y-axis
            ),
            zaxis=dict(
                title='UMAP3',
                showbackground=True  # Remove background plane for z-axis
            )
        ),        
        width=1100,
        height=900,
        legend=dict(x=0, y=1)
    )
    
    # Display the plot
    fig.show()

    # Save the plot as a standalone HTML file
    filename = f'UMAP_{type}_Clusters.html'
    filepath = os.path.join(output_dir, filename)
    pio.write_html(fig, file=filepath, full_html=True)
    print(f'Plot saved to {filepath}')


# ----------------------------------------------------------------------
def plot_cluster_feature_importance(cluster_feature_importances, method, columns=2, target_percentage=0.99):
    num_clusters = len(cluster_feature_importances)
    num_cols = columns
    num_rows = (num_clusters + 1) // num_cols
    
    # Define original feature names and corresponding display names
    label_map = {
        # Group 1
        'label_ODEs': 'ODEs', 'label_PDEs': 'PDEs', 'label_Agent_Based_Models': 'ABMs', 
        'label_Hybrid_Multiscale': 'hybrid/multiscale', 'label_ML_AI': 'ML/AI',
        
        # Group 2
        'label_MR_CT': 'MRI/CT', 'label_DTI': 'DTI', 'label_Biopsies_Histopathology': 'biopsies/histopathology', 
        'label_Liquid_Biopsies': 'liquid biopsies', 'label_Omics': 'omics', 
        'label_PET': 'PET', 'label_Multiphoton_Imaging': 'multiphoton imaging', 'label_Photoacoustics': 'photoacoustics',
        'label_in_vitro_mouse':'pre-clinical',
        
        # Group 3
        'label_group_proliferation': 'proliferation', 'label_group_invasion': 'invasion', 
        'label_group_vasomodulation': 'vasomodulation', 'label_group_evolution': 'evolution', 
        'label_group_epigenetics': 'epigenetics', 'label_group_metabolism': 'metabolism', 
        'label_group_microenvironment': 'microenvironment', 'label_group_phenotypic_plasticity': 'phenotypic plasticity', 
        'label_group_immune': 'immune', 'label_group_biomechanics': 'biomechanics'
        # Group 4 (ungrouped Hallmarks)
        ,'label_cell-cycle':'cell-cycle', 'label_growth_rate':'growth rate', 'label_proliferation':'proliferation', 
        'label_chemotaxis-haptotaxis':'chemotaxis/haptotaxis', 'label_invasion':'invasion', 'label_migration':'migration',
        'label_angiogenesis':'angiogenesis', 'label_vasculature':'vasculature', 'label_BBB':'BBB', 
        'label_carcinogenesis':'carcinogenesis', 'label_clonal_evolution':'clonal evolution', 'label_stem_cells':'stem cells', 'label_tumorigenesis':'tumorigenesis',
        'label_epigenomics_epigenetics':'epigenomics/epigenetics',
        'label_glycolysis':'glycolysis', 'label_metabolism':'metabolism',
        'label_brain_microenvironment':'brain microenvironment', 'label_tumor_microenvironment':'TME', 
        'label_phenotypic_plasticity':'phenotypic plasticity', 
        'label_immune':'immune',
        'label_biomechanical':'biomechanical','label_mechanobiology':'mechanobiology'
    }

    # Group 1, 2, 3 original labels
    group1 = [
        'label_ODEs', 'label_PDEs', 'label_Agent_Based_Models', 'label_Hybrid_Multiscale', 'label_ML_AI'
    ]
    group2 = [
        'label_MR_CT', 'label_DTI', 'label_Biopsies_Histopathology', 'label_Liquid_Biopsies', 'label_Omics',
        'label_PET', 'label_Multiphoton_Imaging', 'label_Photoacoustics', 'label_in_vitro_mouse'
    ]
    group3 = [
        'label_group_proliferation', 'label_group_invasion', 'label_group_vasomodulation',
        'label_group_evolution', 'label_group_epigenetics', 'label_group_metabolism',
        'label_group_microenvironment', 'label_group_phenotypic_plasticity', 'label_group_immune',
        'label_group_biomechanics'
    ]
    group4 = [
        'label_cell-cycle', 'label_growth_rate', 'label_proliferation', 'label_chemotaxis-haptotaxis', 
        'label_invasion', 'label_migration', 'label_angiogenesis', 'label_vasculature', 'label_BBB', 
        'label_carcinogenesis', 'label_clonal_evolution', 'label_stem_cells', 'label_tumorigenesis',
        'label_epigenomics_epigenetics', 'label_glycolysis', 'label_metabolism', 'label_brain_microenvironment',
        'label_tumor_microenvironment', 'label_phenotypic_plasticity', 'label_immune',
        'label_biomechanical','label_mechanobiology'
    ]

    # Assign a color to each group
    color_map = {}
    color_map.update({feature: 'lightskyblue' for feature in group1})
    color_map.update({feature: 'mediumturquoise' for feature in group2})
    color_map.update({feature: 'lightsalmon' for feature in group3})
    color_map.update({feature: 'lightsalmon' for feature in group4})

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows))
    axes = axes.flatten() if num_rows * num_cols > 1 else [axes]

    for i, (cluster, importance_df) in enumerate(cluster_feature_importances.items()):
        ax = axes[i]
        
        # Sort features by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        
        # Calculate cumulative sum of importance scores
        importance_df['Cumulative Sum'] = importance_df['Importance'].cumsum()
        
        # Calculate the percentage contribution of each feature
        total_importance = importance_df['Importance'].sum()
        importance_df['Cumulative Percentage'] = importance_df['Cumulative Sum'] / total_importance
        
        # Filter features to include only those up to the target percentage
        filtered_df = importance_df[importance_df['Cumulative Percentage'] <= target_percentage]

        # Determine bar colors based on feature grouping
        bar_colors = [color_map.get(feature, 'gray') for feature in filtered_df['Feature']]

        # Use the label_map to replace feature names in the plot
        filtered_df['Display Feature'] = filtered_df['Feature'].map(label_map).fillna(filtered_df['Feature'])

        # Plot only the filtered features
        ax.barh(filtered_df['Display Feature'], filtered_df['Importance'], color=bar_colors)
        ax.set_xlabel('Importance Score')
        # ax.set_ylabel('Features')
        ax.set_title(f'Feature Importance for Cluster {cluster} of {method}')
        ax.set_yticks(range(len(filtered_df)))
        ax.set_yticklabels(filtered_df['Display Feature'])
        ax.set_ylim(-0.5, len(filtered_df) - 0.5)
        ax.invert_yaxis()

    # Hide empty subplots
    for i in range(len(cluster_feature_importances), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------

def save_cluster_feature_importance_with_colors(cluster_feature_importances, method, target_percentage=0.99, top=20, output_dir='plots'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define original feature names and corresponding display names
    label_map = {
        # Group 1
        'label_ODEs': 'ODEs', 'label_PDEs': 'PDEs', 'label_Agent_Based_Models': 'ABMs', 
        'label_Hybrid_Multiscale': 'hybrid/multiscale', 'label_ML_AI': 'ML/AI',
        
        # Group 2
        'label_MR_CT': 'MRI/CT', 'label_DTI': 'DTI', 'label_Biopsies_Histopathology': 'biopsies/histopathology', 
        'label_Liquid_Biopsies': 'liquid biopsies', 'label_Omics': 'omics', 
        'label_PET': 'PET', 'label_Multiphoton_Imaging': 'multiphoton imaging', 'label_Photoacoustics': 'photoacoustics',
        'label_in_vitro_mouse':'pre-clinical',
        
        # Group 3
        'label_group_proliferation': 'proliferation', 'label_group_invasion': 'invasion', 
        'label_group_vasomodulation': 'vasomodulation', 'label_group_evolution': 'evolution', 
        'label_group_epigenetics': 'epigenetics', 'label_group_metabolism': 'metabolism', 
        'label_group_microenvironment': 'microenvironment', 'label_group_phenotypic_plasticity': 'phenotypic plasticity', 
        'label_group_immune': 'immune', 'label_group_biomechanics': 'biomechanics'
        # Group 4 (ungrouped Hallmarks)
        ,'label_cell-cycle':'cell-cycle', 'label_growth_rate':'growth rate', 'label_proliferation':'proliferation', 
        'label_chemotaxis-haptotaxis':'chemotaxis/haptotaxis', 'label_invasion':'invasion', 'label_migration':'migration',
        'label_angiogenesis':'angiogenesis', 'label_vasculature':'vasculature', 'label_BBB':'BBB', 
        'label_carcinogenesis':'carcinogenesis', 'label_clonal_evolution':'clonal evolution', 'label_stem_cells':'stem cells', 'label_tumorigenesis':'tumorigenesis',
        'label_epigenomics_epigenetics':'epigenomics/epigenetics',
        'label_glycolysis':'glycolysis', 'label_metabolism':'metabolism',
        'label_brain_microenvironment':'brain microenvironment', 'label_tumor_microenvironment':'TME', 
        'label_phenotypic_plasticity':'phenotypic plasticity', 
        'label_immune':'immune',
        'label_biomechanical':'biomechanical','label_mechanobiology':'mechanobiology'
    }

    # Group 1, 2, 3 original labels
    group1 = [
        'label_ODEs', 'label_PDEs', 'label_Agent_Based_Models', 'label_Hybrid_Multiscale', 'label_ML_AI'
    ]
    group2 = [
        'label_MR_CT', 'label_DTI', 'label_Biopsies_Histopathology', 'label_Liquid_Biopsies', 'label_Omics',
        'label_PET', 'label_Multiphoton_Imaging', 'label_Photoacoustics', 'label_in_vitro_mouse',
    ]
    group3 = [
        'label_group_proliferation', 'label_group_invasion', 'label_group_vasomodulation',
        'label_group_evolution', 'label_group_epigenetics', 'label_group_metabolism',
        'label_group_microenvironment', 'label_group_phenotypic_plasticity', 'label_group_immune',
        'label_group_biomechanics'
    ]
    group4 = [
        'label_cell-cycle', 'label_growth_rate', 'label_proliferation', 'label_chemotaxis-haptotaxis', 
        'label_invasion', 'label_migration', 'label_angiogenesis', 'label_vasculature', 'label_BBB', 
        'label_carcinogenesis', 'label_clonal_evolution', 'label_stem_cells', 'label_tumorigenesis',
        'label_epigenomics_epigenetics', 'label_glycolysis', 'label_metabolism', 'label_brain_microenvironment',
        'label_tumor_microenvironment', 'label_phenotypic_plasticity', 'label_immune',
        'label_biomechanical','label_mechanobiology'
    ]

    # Assign a color to each group
    color_map = {}
    color_map.update({feature: 'lightskyblue' for feature in group1})
    color_map.update({feature: 'mediumturquoise' for feature in group2})
    color_map.update({feature: 'lightsalmon' for feature in group3})
    color_map.update({feature: 'lightsalmon' for feature in group4})

    for cluster, importance_df in cluster_feature_importances.items():
        fig, ax = plt.subplots(figsize=(7, 7))

        # Sort features by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top)

        # Calculate cumulative sum of importance scores
        importance_df['Cumulative Sum'] = importance_df['Importance'].cumsum()

        # Calculate the percentage contribution of each feature
        total_importance = importance_df['Importance'].sum()
        importance_df['Cumulative Percentage'] = importance_df['Cumulative Sum'] / total_importance

        # Filter features to include only those up to the target percentage
        filtered_df = importance_df[importance_df['Cumulative Percentage'] <= target_percentage]

        # Determine bar colors based on feature grouping
        bar_colors = [color_map.get(feature, 'gray') for feature in filtered_df['Feature']]

        # Use the label_map to replace feature names in the plot
        filtered_df['Display Feature'] = filtered_df['Feature'].map(label_map).fillna(filtered_df['Feature'])

        # Plot only the filtered features
        ax.barh(filtered_df['Display Feature'], filtered_df['Importance'], color=bar_colors)
        ax.set_xlabel('Importance Score')
        # ax.set_ylabel('Features')
        ax.set_title(rf'\textbf{{Feature Importance for Cluster {cluster}}}', fontsize=18)
        # ax.set_title(f'Feature Importance for Cluster {cluster}', fontsize=18, fontweight='bold')
        # ax.set_title(f'Feature Importance for Cluster {cluster}', fontsize=18, weight=700)
        # ax.set_title(f'Feature Importance of the ML/AI volume', fontsize=14)
        
        ax.set_yticks(range(len(filtered_df)))
        ax.set_yticklabels(filtered_df['Display Feature'])
        ax.set_ylim(-0.5, len(filtered_df) - 0.5)
        ax.invert_yaxis()

        plt.tight_layout()

        # Save the individual figure with a descriptive name
        file_path = os.path.join(output_dir, f'feature_importance_cluster_{cluster}_{method}.pdf')

        fig.savefig(file_path, dpi=400, bbox_inches='tight')
        plt.close(fig)

    print(f'All cluster feature importance plots have been saved to \"{output_dir}\"')

# ----------------------------------------------------------------------
def plot_umap_clusters_with_filter(data_scaled, data_original, year_range, type, output_dir, n_neighbors=15, min_dist=0.01, n_components=3):
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the UMAP reducer
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    
    # Fit and transform the data
    data_umap = reducer.fit_transform(data_scaled)
    
    # Create a 3D figure object
    fig = go.Figure()

    # Determine cluster type
    clustertype = f'{type}_Cluster'
    
    # Identify unique clusters and sort them
    clusters = sorted(data_original[clustertype].unique())
    
    # Loop through each sorted cluster and add separate traces
    for cluster in clusters:
        cluster_data = data_original[data_original[clustertype] == cluster]
        indices = cluster_data.index
        cluster_umap = data_umap[indices]
        
        # Add the trace
        fig.add_trace(go.Scatter3d(
            x=cluster_umap[:, 0],
            y=cluster_umap[:, 1],
            z=cluster_umap[:, 2],
            mode='markers',
            marker=dict(
                size=cluster_data[year_range],
                opacity=0.7
            ),
            name=f'Cluster {cluster}',  # Name each trace with the cluster number
            text=[f'<b>Cluster {cluster}</b> <br>{title}<br>{authors}<br>{year}' for year, title, authors in zip(cluster_data['Year'], cluster_data['Title'], cluster_data['Authors'])],
            hovertemplate='%{text}<br><extra></extra>'  # Display custom hover text
        ))
    
    # Update the layout of the figure
    fig.update_layout(
        title=dict(
            text=f"UMAP Projection using {len(clusters)} {type} Clusters - {year_range}",
            font=dict(family="Times New Roman", size=30)
        ),
        scene=dict(
            xaxis=dict(title='UMAP 1', titlefont=dict(family="Times New Roman", size=20)),
            yaxis=dict(title='UMAP 2', titlefont=dict(family="Times New Roman", size=20)),
            zaxis=dict(title='UMAP 3', titlefont=dict(family="Times New Roman", size=20))
        ),
        width=1100,
        height=900,
        legend=dict(
            x=0, y=1,
            font=dict(family="Times New Roman", size=18)
        )
    )
    
    # Display the plot
    fig.show()

# ----------------------------------------------------------------------
def plot_umap_clusters_with_filter_and_save(data_scaled, data_original, year_range, cluster_type, n_neighbors=15, min_dist=0.01, n_components=3, output_dir='plots'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    title_font_size = 30
    label_font_size = 20 
    legend_font_size = 20
    
    # Initialize the UMAP reducer
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)

    # Fit and transform the data
    data_umap = reducer.fit_transform(data_scaled)

    # Create a 3D figure object
    fig = go.Figure()

    # Construct the cluster column name
    cluster_col = f'{cluster_type}_Cluster'

    # Identify unique clusters and sort them
    clusters = sorted(data_original[cluster_col].unique())

    # Loop through each sorted cluster and add separate traces
    for cluster in clusters:
        cluster_data = data_original[data_original[cluster_col] == cluster]
        indices = cluster_data.index
        cluster_umap = data_umap[indices]

        # Add the trace
        fig.add_trace(go.Scatter3d(
            x=cluster_umap[:, 0],
            y=cluster_umap[:, 1],
            z=cluster_umap[:, 2],
            mode='markers',
            marker=dict(
                size=cluster_data[year_range],
                opacity=0.7
            ),
            name=f'Cluster {cluster}',
            text=cluster_data['Year']
        ))

    # Update the layout of the figure
    fig.update_layout(
        title=dict(
            text=f"UMAP Projection using {len(clusters)} {cluster_type} Clusters - {year_range}",
            font=dict(size=title_font_size, family='Times New Roman')
        ),
        scene=dict(
            xaxis=dict(title='UMAP 1', titlefont=dict(size=label_font_size, family='Times New Roman')),
            yaxis=dict(title='UMAP 2', titlefont=dict(size=label_font_size, family='Times New Roman')),
            zaxis=dict(title='UMAP 3', titlefont=dict(size=label_font_size, family='Times New Roman'))
        ),
        width=1100,
        height=900,
        legend=dict(
            x=0, y=1,
            font=dict(size=legend_font_size, family='Times New Roman')
        )
    )

    # Save the plot to an image file
    filename = f'UMAP_{cluster_type}_Clusters_{year_range}.pdf'
    filepath = os.path.join(output_dir, filename)
    fig.write_image(filepath)
    print(f'Plot saved to {filepath}')



# ----------------------------------------------------------------------
def add_opacity_columns(data_original, year_ranges):
    for start, end in year_ranges:
        column_name = f'{start}-{end}'
        data_original[column_name] = (
            (data_original['Year'] >= start) & (data_original['Year'] <= end)
        ).astype(int)
    return data_original
    

# ----------------------------------------------------------------------
def get_titles_and_authors_by_cluster(dataset,cluster_type, cluster_number):
    if cluster_type not in ['Kmeans_Cluster', 'GMM_Cluster']:
        raise ValueError("Invalid cluster type. Choose 'Kmeans_Cluster' or 'GMM_Cluster'.")
    
    # Ensure the dataset has an 'Authors' column
    if 'Authors' not in dataset.columns:
        raise ValueError("The dataset does not contain an 'Authors' column.")
    
    # Group titles and authors by the specified cluster type
    cluster_data = dataset.groupby(cluster_type).apply(lambda x: list(zip(x['Title'], x['Authors']))).reset_index()
    cluster_data.columns = [cluster_type, 'Titles_Authors']
    
    # Get the titles and authors for the specified cluster number
    titles_authors = cluster_data[cluster_data[cluster_type] == cluster_number]['Titles_Authors']
    
    if titles_authors.empty:
        return f"No titles or authors found for {cluster_type} = {cluster_number}"
    
    return titles_authors.iloc[0]

# ----------------------------------------------------------------------
def get_titles_authors_years_by_cluster(dataset, cluster_type, cluster_number):
    if cluster_type not in ['Kmeans_Cluster', 'GMM_Cluster']:
        raise ValueError("Invalid cluster type. Choose 'Kmeans_Cluster' or 'GMM_Cluster'.")
    
    # Ensure the dataset has 'Authors' and 'Year' columns
    if 'Authors' not in dataset.columns or 'Year' not in dataset.columns:
        raise ValueError("The dataset does not contain 'Authors' or 'Year' columns.")
    
    # Group titles, authors, and years by the specified cluster type
    cluster_data = dataset.groupby(cluster_type).apply(lambda x: list(zip(x['Title'], x['Authors'], x['Year']))).reset_index()
    cluster_data.columns = [cluster_type, 'Titles_Authors_Years']
    
    # Get the titles, authors, and years for the specified cluster number
    titles_authors_years = cluster_data[cluster_data[cluster_type] == cluster_number]['Titles_Authors_Years']
    
    if titles_authors_years.empty:
        return f"No titles, authors, or years found for {cluster_type} = {cluster_number}"
    
    return titles_authors_years.iloc[0]

# ----------------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier


# ----------------------------------------------------------------------
# Function to classify each group and calculate a single probability for the entire target group
def classify_group(dataset, features_to_use, target_group, group_name):
    X = dataset[features_to_use]  # Features from the other two groups
    y = dataset[target_group]  # Target group

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a multi-label classifier (for multiple binary labels)
    clf = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    clf.fit(X_scaled, y)

    # Predict probabilities on the test set
    y_probs = clf.predict_proba(X_scaled)

    # Calculate an overall probability for the group by averaging the probabilities of each label
    overall_probs = np.mean([np.array([p[1] for p in prob]) for prob in y_probs], axis=0)

    # Convert the overall probability to a DataFrame
    probs = pd.DataFrame({f"{group_name}_prob": overall_probs}, index=dataset.index)

    return probs

# ----------------------------------------------------------------------
def classify_group_all(dataset, features_to_use, target_group, group_name):
    X = dataset[features_to_use]  # Features from the other two groups
    y = dataset[target_group]  # Target group

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train a multi-label classifier (for multiple binary labels)
    clf = MultiOutputClassifier(RandomForestClassifier(random_state=42))
    clf.fit(X_scaled, y)

    # Predict probabilities on the test set
    y_probs = clf.predict_proba(X_scaled)

    # Calculate an overall probability for the group by averaging the probabilities of each label
    # overall_probs = np.mean([np.array([p[1] for p in prob]) for prob in y_probs], axis=0)
    overall_probs = [np.array([p[1] for p in prob]) for prob in y_probs]

    # Convert the overall probability to a DataFrame
    # probs = pd.DataFrame({f"{group_name}_prob": overall_probs}, index=dataset.index)

    return overall_probs

# ----------------------------------------------------------------------
import plotly.graph_objs as go
import plotly.io as pio
import os

def plot_clustered_probabilities_with_centers(data, prob_cols=['MM', 'DT', 'Hm'], output_dir=None, cluster_col=None):
    # output_dir = output_dir
    if output_dir is None:
        output_dir = "results"
    if cluster_col is None:
        cluster_col = f"{method}_Cluster" 

    os.makedirs(output_dir, exist_ok=True)

    # Custom colors for the clusters
    custom_colors = [
        '#636efa', '#ef553b', '#00cc96', '#ffa15a','#ab63fa', 
        '#1bd3f3', '#ff6c96', '#cbeea5', '#fe9eff'
    ]

    # Create a 3D figure object
    fig = go.Figure()

    # Identify unique clusters and sort them
    clusters = sorted(data[cluster_col].unique())

    # Ensure that the number of clusters doesn't exceed the number of custom colors
    if len(clusters) > len(custom_colors):
        raise ValueError("The number of clusters exceeds the number of custom colors provided.")

    # Loop through each sorted cluster and add separate traces for data points
    for idx, cluster in enumerate(clusters):
        cluster_data = data[data[cluster_col] == cluster]
        cluster_color = custom_colors[idx]  # Assign a color from custom_colors
        
        # Add the trace for each cluster's data points
        fig.add_trace(go.Scatter3d(
            x=cluster_data[prob_cols[0]],  # MM
            y=cluster_data[prob_cols[1]],  # DT
            z=cluster_data[prob_cols[2]],  # Hm
            mode='markers',
            marker=dict(
                size=(cluster_data['Year'] - cluster_data['Year'].min()) / (cluster_data['Year'].max() - cluster_data['Year'].min()) * 20,
                color=cluster_color,  # Use the custom color for the cluster
                opacity=0.7
            ),
            name=f'Cluster {cluster}',  # Name each trace with the cluster number
            text=[f'<b>Cluster {cluster}</b> <br>{title}<br>{authors}<br>{year}' for year, title, authors in zip(cluster_data['Year'], cluster_data['Title'], cluster_data['Authors'])],
            hovertemplate='%{text}<br><extra></extra>'  # Display custom hover text
        ))

    # Calculate cluster centers (mean 'MM', 'DT', and 'Hm' for each cluster)
    cluster_centers = data.groupby(cluster_col)[prob_cols].mean()

    # Add the trace for the cluster centers using the same custom color and black outline
    for idx, cluster in enumerate(clusters):
        fig.add_trace(go.Scatter3d(
            x=[cluster_centers.loc[cluster, prob_cols[0]]],  # MM for center
            y=[cluster_centers.loc[cluster, prob_cols[1]]],  # DT for center
            z=[cluster_centers.loc[cluster, prob_cols[2]]],  # Hm for center
            mode='markers',
            marker=dict(
                size=15,  # Larger size for cluster centers
                color=custom_colors[idx],  # Same custom color as the cluster
                symbol='diamond',  # Use diamond shape for centers
                opacity=1,  # Full opacity for centers
                line=dict(
                    color='black',  # Black outline
                    width=2  # Thickness of the outline
                )
            ),
            name=f'Center of Cluster {cluster}',
            text=[f'<b>Center of Cluster {cluster}</b>'],
            hovertemplate='%{text}<br>MM: %{x}<br>DT: %{y}<br>Hm: %{z}<extra></extra>'  # Hover text for centers
        ))

    # Update the layout of the figure, ensuring equal aspect ratio
    fig.update_layout(
        title="Cluster projection based on multiclass classification probability",
        scene=dict(
            xaxis_title='Mathematical Methods',
            yaxis_title='Data-Generating Technologies',
            zaxis_title='Hallmarks',
            aspectmode='cube',  # Makes the ratio equal between axes
        ),
        width=1100,
        height=900,
        legend=dict(x=0, y=1)
    )

    # Display the plot
    fig.show()

    # Save the plot as a standalone HTML file
    filename = '3D_Probability_Clusters_with_Custom_Centers.html'
    filepath = os.path.join(output_dir, filename)
    pio.write_html(fig, file=filepath, full_html=True)
    print(f'Plot saved to {filepath}')

# ----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# Function to plot the distribution for a specific cluster and axis using seaborn's histplot
def plot_cluster_distribution_sns(data, method, cluster_num, axes):
    # Define axis labels
    axis_labels = {
        'MM': 'Mathematical Methods',
        'DT': 'Data-Generating Technologies',
        'Hm': 'Hallmarks'
    }
    custom_colors = [
        '#1bd3f3', '#636efa', '#ef553b', '#00cc96', '#ffa15a','#ab63fa', 
        '#1bd3f3', '#ff6c96', '#cbeea5', '#fe9eff'
    ]

    # Loop through each axis and plot the distribution
    for i, axis in enumerate(['MM', 'DT', 'Hm']):
        # Filter the data for the specified cluster
        cluster_data = data[data[f'{method}_Cluster'] == cluster_num]
        
        # Create the KDE plot on the current axis
        sns.kdeplot(
            cluster_data[axis],
            color=custom_colors[cluster_num % len(custom_colors)],  # Use the custom color based on cluster number
            fill=True,  # Optional: Fill the area under the KDE curve
            bw_adjust=0.4,  # Adjust bandwidth for smoother KDE curve (optional)
            ax=axes[i],  # Plot on the appropriate subplot axis
            alpha=0.3  # Set transparency for the plot itself
        )

        # Set titles and labels for each subplot
        # axes[i].set_title(f"Distribution on {axis_labels[axis]}", fontsize=14)
        axes[i].set_xlabel(f'{axis_labels[axis]} Probability', fontsize=20)
        axes[i].set_ylabel('Density', fontsize=20) if i==0 else axes[i].set_ylabel('', fontsize=10)

        # Create a custom legend inside each subplot with transparency (alpha)
        custom_line = mlines.Line2D([], [], color=custom_colors[cluster_num % len(custom_colors)], 
                                    label=f'Cluster {cluster_num}', linewidth=5, alpha=0.5)
        axes[2].legend(handles=[custom_line], loc='upper right', fontsize=18, handlelength=1)  # Adjust line length with handlelength


# ----------------------------------------------------------------------
#
# Everything below this line was migrated out of Brain_Cancer_Survey.ipynb so
# that the notebook can stay short and simply call into this module. Each
# function below is a direct, behavior-preserving transcription of the code
# that used to live inline in a specific notebook cell (noted in the
# docstring/comment above each function).
#
# ----------------------------------------------------------------------

import matplotlib.ticker as ticker
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap, to_rgba, to_hex
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
import subprocess


# ----------------------------------------------------------------------
# Shared constants
# ----------------------------------------------------------------------

# Full feature list used both for UMAP (notebook cell "Choose features from
# original dataset") and for the cluster feature-importance analysis.
ALL_FEATURES = [
    'label_ODEs', 'label_PDEs', 'label_Agent_Based_Models', 'label_Hybrid_Multiscale', 'label_ML_AI',
    'label_MR_CT', 'label_DTI', 'label_Biopsies_Histopathology', 'label_Liquid_Biopsies',
    'label_Omics', 'label_PET', 'label_Multiphoton_Imaging', 'label_Photoacoustics', 'label_in_vitro_mouse',
    'label_cell-cycle', 'label_growth_rate', 'label_proliferation',
    'label_chemotaxis-haptotaxis', 'label_invasion', 'label_migration',
    'label_angiogenesis', 'label_vasculature', 'label_BBB',
    'label_carcinogenesis', 'label_clonal_evolution', 'label_stem_cells', 'label_tumorigenesis',
    'label_epigenomics_epigenetics',
    'label_glycolysis', 'label_metabolism',
    'label_brain_microenvironment', 'label_tumor_microenvironment',
    'label_phenotypic_plasticity',
    'label_immune',
    'label_biomechanical', 'label_mechanobiology'
]

# Three feature groups used for the multiclass classification / MM-DT-Hm scores
MATH_METHODS = ['label_ODEs', 'label_PDEs', 'label_Agent_Based_Models', 'label_Hybrid_Multiscale', 'label_ML_AI']
DATA_TECHNOLOGIES = ['label_MR_CT', 'label_DTI', 'label_Biopsies_Histopathology', 'label_Liquid_Biopsies', 'label_Omics',
                      'label_PET', 'label_Multiphoton_Imaging', 'label_Photoacoustics', 'label_in_vitro_mouse']
HALLMARKS = ['label_cell-cycle', 'label_growth_rate', 'label_proliferation', 'label_chemotaxis-haptotaxis', 'label_invasion',
             'label_migration', 'label_angiogenesis', 'label_vasculature', 'label_BBB', 'label_carcinogenesis',
             'label_clonal_evolution', 'label_stem_cells', 'label_tumorigenesis', 'label_epigenomics_epigenetics',
             'label_glycolysis', 'label_metabolism', 'label_brain_microenvironment', 'label_tumor_microenvironment',
             'label_phenotypic_plasticity', 'label_immune', 'label_biomechanical', 'label_mechanobiology']
REST_FEATURES = ['Year', 'label_Mathematical_Methods', 'label_Data_Technologies', 'label_Hallmarks']

# 4-5-year publication windows used throughout the notebook
YEAR_RANGES = [(2000, 2004), (2005, 2009), (2010, 2013), (2014, 2017), (2018, 2021), (2022, 2025)]

# Colors used by the cluster-over-time streamgraphs (same values as the
# module-level `custom_colors` above; kept as a separate name here to match
# how each notebook section originally scoped its own palette)
CLUSTER_STREAM_COLORS = [
    '#636efa', '#ef553b', '#00cc96', '#ffa15a', '#ab63fa',
    '#1bd3f3', '#ff6c96', '#cbeea5', '#fe9eff'
]

# Colors used by the 3D probability-density / trajectory plots
TRAJECTORY_COLORS = [
    '#1bd3f3', '#636efa', '#ef553b', '#00cc96', '#ffa15a', '#ab63fa',
    '#1bd3f3', '#ff6c96', '#cbeea5', '#fe9eff'
]

CLUSTER_DESCRIPTIONS = [
    "Continuum Image-informed Models",     # Cluster 1
    "Predictive ML \& Physics-Based",      # Cluster 2
    "Mechanistic Progression \& Therapy",  # Cluster 3
    "Differential Invasion Models",        # Cluster 4
    "Cluster 5",
    "Cluster 6",
    "Cluster 7",
    "Cluster 8",
    "Cluster 9"
]


# ----------------------------------------------------------------------
# General setup / data loading
# ----------------------------------------------------------------------

def set_plot_style(font_size=18, usetex=True, font_family='Times New Roman'):
    """Apply the notebook's standard matplotlib style (used repeatedly throughout)."""
    plt.rcParams["text.usetex"] = usetex
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size


def load_dataset(data_dir, filename='Main_Dataset.csv', encoding='utf-8'):
    """Load the main survey dataset (notebook: 'Set Dataset')."""
    return pd.read_csv(os.path.join(data_dir, filename), encoding=encoding)


def split_datasets_by_method(data_org):
    """Split the dataset into per-mathematical-method subsets (notebook: 'Split to datasets of Mathematical Methods')."""
    return {
        'ODEs': data_org.loc[data_org['label_ODEs'] == 1].reset_index(drop=True),
        'PDEs': data_org.loc[data_org['label_PDEs'] == 1].reset_index(drop=True),
        'Agent_based': data_org.loc[data_org['label_Agent_Based_Models'] == 1].reset_index(drop=True),
        'Hybrid_Multiscale': data_org.loc[data_org['label_Hybrid_Multiscale'] == 1].reset_index(drop=True),
        'ML_AI': data_org.loc[data_org['label_ML_AI'] == 1].reset_index(drop=True),
    }


# ----------------------------------------------------------------------
# Histograms / streamgraphs of label volume over time
# ----------------------------------------------------------------------

def plot_methods_streamgraph(data, output_dir, percentage=False):
    """Streamgraph of mathematical-methods labels per year."""
    methods_columns = [
        'label_ODEs', 'label_PDEs', 'label_Agent_Based_Models',
        'label_Hybrid_Multiscale', 'label_ML_AI'
    ]
    methods_data = data[['Year'] + methods_columns]
    methods_per_year = methods_data.groupby('Year').sum() / data.shape[0] * 100 if percentage else methods_data.groupby('Year').sum()

    colors = plt.cm.Blues([0.2, 0.4, 0.6, 0.8, 1.0])
    custom_labels_MM = ['ODEs', 'PDEs', 'ABMs', 'hybrid/multiscale', 'ML/AI']

    plt.figure(figsize=(7, 5))
    plt.stackplot(
        methods_per_year.index,
        methods_per_year.T,
        labels=custom_labels_MM,
        colors=colors,
        alpha=0.8,
    )
    plt.rcParams["text.usetex"] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    plt.legend(custom_labels_MM, title='Mathematical Methods', loc="upper left", ncol=1, fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('\% of total volume') if percentage else plt.ylabel('Number of Papers')

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xlim(2000, 2025)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])

    plt.tight_layout()
    if percentage:
        plt.savefig(f'{output_dir}/Streamgraph_mathematical_methods_percent_vs_time.pdf', dpi=400, bbox_inches='tight')
    else:
        plt.savefig(f'{output_dir}/Streamgraph_mathematical_methods_vs_time.pdf', dpi=400, bbox_inches='tight')
    plt.show()

    return methods_per_year


def plot_data_technologies_streamgraph(data, output_dir, percentage=False):
    """Streamgraph of data-generating-technology labels per year."""
    custom_data_tech_columns = [
        'label_MR_CT', 'label_DTI', 'label_Biopsies_Histopathology',
        'label_Liquid_Biopsies', 'label_Omics', 'label_PET',
        'label_Multiphoton_Imaging', 'label_Photoacoustics', 'label_in_vitro_mouse'
    ]
    custom_data_tech_data = data[['Year'] + custom_data_tech_columns]
    custom_data_tech_per_year = custom_data_tech_data.groupby('Year').sum() / data.shape[0] * 100 if percentage else custom_data_tech_data.groupby('Year').sum()

    colors = plt.cm.Greens([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    plt.figure(figsize=(7, 5))
    plt.stackplot(
        custom_data_tech_per_year.index,
        custom_data_tech_per_year.T,
        labels=custom_data_tech_columns,
        colors=colors,
        alpha=0.8
    )
    plt.rcParams["text.usetex"] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    custom_labels_DT = [
        'MR/CT', 'DTI', 'biopsies/histopathology',
        'liquid biopsies', 'omics', 'PET',
        'multiphoton imaging', 'photoacoustics', 'pre-clinical'
    ]
    plt.legend(custom_labels_DT, title='Data-Generating Technologies', loc="upper left", ncol=1, fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('\% of total volume') if percentage else plt.ylabel('Number of Papers')

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xlim(2000, 2025)

    plt.tight_layout()
    if percentage:
        plt.savefig(f'{output_dir}/Streamgraph_data_technologies_percent_vs_time.pdf', dpi=400, bbox_inches='tight')
    else:
        plt.savefig(f'{output_dir}/Streamgraph_data_technologies_vs_time.pdf', dpi=400, bbox_inches='tight')
    plt.show()

    return custom_data_tech_per_year


def plot_hallmark_groups_streamgraph(data, output_dir, percentage=False):
    """Streamgraph of Hallmark-group labels per year."""
    hallmark_group_columns = [
        'label_group_proliferation', 'label_group_invasion', 'label_group_vasomodulation',
        'label_group_evolution', 'label_group_epigenetics', 'label_group_metabolism',
        'label_group_microenvironment', 'label_group_phenotypic_plasticity',
        'label_group_immune', 'label_group_biomechanics'
    ]
    hallmark_group_data = data[['Year'] + hallmark_group_columns]
    hallmark_group_per_year = hallmark_group_data.groupby('Year').sum() / data.shape[0] * 100 if percentage else hallmark_group_data.groupby('Year').sum()

    colors_orange = plt.cm.Oranges([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    plt.figure(figsize=(7, 5))
    plt.stackplot(
        hallmark_group_per_year.index,
        hallmark_group_per_year.T,
        labels=hallmark_group_columns,
        colors=colors_orange,
        alpha=0.8
    )
    plt.rcParams["text.usetex"] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18
    custom_labels_H = [
        'proliferation', 'invasion', 'vasomodulation',
        'evolution', 'epigenetics', 'metabolism',
        'microenvironment', 'phenotypic plasticity',
        'immune', 'biomechanics'
    ]
    plt.legend(custom_labels_H, title='Hallmark Groups', loc="upper left", ncol=1, fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('\% of total volume') if percentage else plt.ylabel('Number of Papers')

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xlim(2000, 2025)

    plt.tight_layout()
    if percentage:
        plt.savefig(f'{output_dir}/Streamgraph_hallmark_groups_percent_vs_time.pdf', dpi=400, bbox_inches='tight')
    else:
        plt.savefig(f'{output_dir}/Streamgraph_hallmark_groups_vs_time.pdf', dpi=400, bbox_inches='tight')
    plt.show()

    return hallmark_group_per_year


def plot_total_papers_histogram(data, output_dir, percentage=False):
    """Bar chart of total publication volume per year."""
    total_papers_per_year = data.groupby('Year').size() / data.shape[0] * 100 if percentage else data.groupby('Year').size()

    plt.figure(figsize=(7, 5))
    plt.rcParams["text.usetex"] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    plt.bar(
        total_papers_per_year.index,
        total_papers_per_year.values,
        color='#6c97a5',
        edgecolor='white',
        alpha=0.8,
        width=1
    )

    plt.xlabel('Year')
    plt.ylabel('\% of total volume') if percentage else plt.ylabel('Number of Papers')

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xlim(1999, 2026)
    ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2025])

    plt.tight_layout()
    if percentage:
        plt.savefig(f'{output_dir}/Histogram_total_papers_percent_vs_time.pdf', dpi=400, bbox_inches='tight')
    else:
        plt.savefig(f'{output_dir}/Histogram_total_papers_vs_time.pdf', dpi=400, bbox_inches='tight')
    plt.show()

    return total_papers_per_year


# ----------------------------------------------------------------------
# UMAP feature preparation / clustering
# ----------------------------------------------------------------------

def prepare_features_for_umap(data, features=None):
    """Scale the chosen feature columns ahead of UMAP (notebook: 'Choose features from original dataset')."""
    if features is None:
        features = ALL_FEATURES
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    data_scaled = X_scaled.copy()
    data_scaled_for_Umap = data_scaled.copy()
    data_for_Umap = data[features].copy()
    return X, X_scaled, data_scaled, data_scaled_for_Umap, data_for_Umap


def compute_umap_embedding(data, data_for_Umap, n_neighbors=30, min_dist=0.01, n_components=3, random_state=42):
    """Fit UMAP, attach UMAP1-3 to `data`, and re-scale on the UMAP axes (notebook: 'Cluster on UMAP features')."""
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=random_state)
    data_umap = reducer.fit_transform(data_for_Umap)

    data[['UMAP1', 'UMAP2', 'UMAP3']] = data_umap
    features = ['UMAP1', 'UMAP2', 'UMAP3']
    X = data[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    data_scaled = X_scaled.copy()

    return data, features, X, X_scaled, data_scaled


def plot_umap_3d_scatter(data):
    """3D scatter of the UMAP1-3 embedding."""
    fig = go.Figure(data=[go.Scatter3d(
        x=data['UMAP1'],
        y=data['UMAP2'],
        z=data['UMAP3'],
        mode='markers',
        marker=dict(
            size=5,
            colorscale='Viridis',
            opacity=0.8
        )
    )])

    fig.update_layout(
        title="3D Scatter Plot of the First Three Principal Components",
        scene=dict(
            xaxis=dict(title='UMAP1', showbackground=False),
            yaxis=dict(title='UMAP2', showbackground=False),
            zaxis=dict(title='UMAP3', showbackground=True)
        ),
        width=900,
        height=800
    )

    fig.show()
    return fig


def run_kmeans_clustering(data, X_scaled, features, k=4, random_state=42):
    """K-means clustering on the (UMAP-derived) scaled features."""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
    Kmeans_clusters = kmeans.fit_predict(X_scaled)

    data.loc[:, 'Kmeans_Cluster'] = Kmeans_clusters
    cluster_means = data.groupby('Kmeans_Cluster')[features].mean()

    sil_score = silhouette_score(X_scaled, Kmeans_clusters)
    print(f'Silhouette Score: {sil_score:.3f}')

    return data, cluster_means, sil_score


def run_gmm_clustering(data, X_scaled, n_components=4, random_state=42):
    """Gaussian Mixture Model clustering on the (UMAP-derived) scaled features."""
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(X_scaled)
    GMM_labels = gmm.predict(X_scaled)

    data['GMM_Cluster'] = GMM_labels

    sil_score = silhouette_score(X_scaled, GMM_labels)
    print(f'Silhouette Score: {sil_score:.3f}')

    return data, sil_score


def finalize_cluster_labels(data):
    """Shift both cluster label columns so numbering starts at 1 instead of 0."""
    data['Kmeans_Cluster'] = data['Kmeans_Cluster'] + 1
    data['GMM_Cluster'] = data['GMM_Cluster'] + 1
    return data


# ----------------------------------------------------------------------
# Cluster feature importance
# ----------------------------------------------------------------------

def get_feature_importance_features(include_AI, features=None):
    """Feature list for the feature-importance analysis, optionally dropping ML/AI."""
    features = list(ALL_FEATURES) if features is None else list(features)
    if not include_AI and 'label_ML_AI' in features:
        features.remove('label_ML_AI')
    return features


def analyze_cluster_feature_importance(X, y, method, output_dir, save_target_percentage=0.99, save_top=20,
                                        plot_target_percentage=0.84, plot_columns=2):
    """Compute cluster feature importances, save the per-cluster figures, and display the combined overview."""
    cluster_feature_importances = compute_cluster_feature_importances(X, y)
    save_cluster_feature_importance_with_colors(
        cluster_feature_importances, method, output_dir=output_dir,
        target_percentage=save_target_percentage, top=save_top
    )
    plot_cluster_feature_importance(
        cluster_feature_importances, method, columns=plot_columns, target_percentage=plot_target_percentage
    )
    return cluster_feature_importances


# ----------------------------------------------------------------------
# Cluster export / cluster-over-time streamgraphs
# ----------------------------------------------------------------------

def save_publication_lists_by_cluster(data, method, cluster_count, data_dir):
    """Save one CSV of publications per cluster."""
    column_name = f"{method}_Cluster"
    save_path = f"{data_dir}/publication_lists_of_{cluster_count}_{method}_clusters"
    os.makedirs(save_path, exist_ok=True)

    for c in range(1, cluster_count + 1):
        df_c = data[data[column_name] == c]
        file_name = f"{cluster_count}_{method}_Cluster_{c}.csv"
        df_c.to_csv(f"{save_path}/{file_name}", index=False)

    print(f'The publication lists per cluster were saved in {save_path}')
    return save_path


def plot_cluster_streamgraph_by_year(data, method, output_dir, percentage=True):
    """Streamgraph of paper counts per cluster, per calendar year."""
    cluster_counts_per_year_updated = (
        data.groupby(['Year', f'{method}_Cluster']).size().unstack(fill_value=0) / data.shape[0] * 100
        if percentage
        else data.groupby(['Year', 'Kmeans_Cluster']).size().unstack(fill_value=0)
    )

    plt.figure(figsize=(7, 5))
    plt.stackplot(
        cluster_counts_per_year_updated.index,
        cluster_counts_per_year_updated.T,
        colors=CLUSTER_STREAM_COLORS,
        alpha=0.7
    )

    custom_labels = [f"{desc}" for desc in CLUSTER_DESCRIPTIONS]

    plt.rcParams["text.usetex"] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    plt.legend(custom_labels, loc='upper left', ncol=1, fontsize=14)
    plt.xlabel('Year')
    plt.ylabel('\% of total volume') if percentage else plt.ylabel('Number of Papers')

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    plt.xlim(2000, 2025)
    plt.tight_layout()

    if percentage:
        plt.savefig(f'{output_dir}/papers_percent_per_cluster_vs_time.pdf', dpi=400, bbox_inches='tight')
    else:
        plt.savefig(f'{output_dir}/papers_per_cluster_vs_time.pdf', dpi=400, bbox_inches='tight')
    plt.show()

    return cluster_counts_per_year_updated


def plot_cluster_streamgraph_5yr_windows(data, method, output_dir, percentage=False):
    """Symmetric streamgraph of paper counts per cluster, binned into 4-5 year windows, with count annotations."""
    bins = [2000, 2005, 2010, 2014, 2018, 2022, 2026]
    data['Year_Binned'] = pd.cut(
        data['Year'],
        bins=bins,
        right=False,
        labels=['2000–2004', '2005–2009', '2010–2013', '2014–2017', '2018–2021', '2022–2025']
    )

    cluster_counts_per_window = (
        data.groupby(['Year_Binned', f'{method}_Cluster'])
        .size()
        .unstack(fill_value=0)
    )

    cluster_counts_for_plot = (
        cluster_counts_per_window
        .div(cluster_counts_per_window.sum(axis=1), axis=0) * 100
        if percentage
        else cluster_counts_per_window
    )

    window_labels = ["2000", "2005", "2010", "2015", "2020", "2025"]
    cluster_counts_for_plot.index = window_labels

    plt.figure(figsize=(7, 5))
    plt.stackplot(
        cluster_counts_for_plot.index,
        cluster_counts_for_plot.T,
        colors=CLUSTER_STREAM_COLORS,
        alpha=0.7,
        baseline='sym'
    )

    custom_labels = [desc for desc in CLUSTER_DESCRIPTIONS]

    plt.rcParams["text.usetex"] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 14

    plt.legend(
        custom_labels,
        loc='upper left',
        ncol=1,
        fontsize=13,
        frameon=True
    )
    plt.xlabel('Year')

    plt.yticks([])
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)

    def annotate_bin(bin_counts, bin_index, x_offset, h_align, color_list):
        """Annotate the symmetrical stack at `bin_index` with absolute cluster counts."""
        total_in_bin = bin_counts.sum()
        current_lower = -total_in_bin / 2.0
        min_separation = 1.4
        last_label_y = None

        for i, count in enumerate(bin_counts):
            current_upper = current_lower + count
            y_mid = (current_lower + current_upper) / 2.0

            if last_label_y is not None and (y_mid - last_label_y) < min_separation:
                y_mid = last_label_y + min_separation

            plt.text(
                bin_index + x_offset,
                y_mid,
                str(int(count)),
                ha=h_align,
                va='center',
                fontsize=14,
                color=color_list[i]
            )

            last_label_y = y_mid
            current_lower = current_upper

    if not percentage:
        first_bin_counts = cluster_counts_per_window.iloc[0, :].values
        annotate_bin(
            bin_counts=first_bin_counts,
            bin_index=0,
            x_offset=-0.05,
            h_align='right',
            color_list=CLUSTER_STREAM_COLORS
        )

        last_bin_counts = cluster_counts_per_window.iloc[-1, :].values
        last_bin_index = len(window_labels) - 1.12
        annotate_bin(
            bin_counts=last_bin_counts,
            bin_index=last_bin_index,
            x_offset=0.13,
            h_align='left',
            color_list=CLUSTER_STREAM_COLORS
        )

    fig = plt.gcf()
    fig.set_size_inches(7, 5)
    plt.ylabel('\% of papers' if percentage else 'Number of Papers', labelpad=15)
    plt.xlim(-0.2, len(window_labels) - 0.7)
    plt.subplots_adjust(left=0.01, right=0.85, top=0.755, bottom=0.)

    plt.rcParams["text.usetex"] = True
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 18

    if percentage:
        plt.savefig(
            f'{output_dir}/papers_percent_per_cluster_vs_5year_windows_streamgraph.pdf',
            bbox_inches='tight',
            dpi=300
        )
    else:
        plt.savefig(
            f'{output_dir}/papers_per_cluster_vs_5year_windows_streamgraph.pdf',
            bbox_inches='tight',
            dpi=300
        )
    plt.show()

    return cluster_counts_per_window


# ----------------------------------------------------------------------
# Cluster projection on MM-DT-Hallmarks via multiclass classification
# ----------------------------------------------------------------------

def get_classification_feature_groups(include_AI=True):
    """Return (math_methods, data_technologies, hallmarks, rest) feature-group lists."""
    math_methods = MATH_METHODS.copy()
    data_technologies = DATA_TECHNOLOGIES.copy()
    hallmarks = HALLMARKS.copy()
    rest = REST_FEATURES.copy()
    if not include_AI and 'label_ML_AI' in math_methods:
        math_methods.remove('label_ML_AI')
    return math_methods, data_technologies, hallmarks, rest


def classify_group_probabilities(dataset, math_methods, data_technologies, hallmarks, rest):
    """Per-group overall classification probability (one probability per publication per group)."""
    features_math = data_technologies + hallmarks + rest
    math_probs = classify_group(dataset, features_math, math_methods, 'math')

    features_data_tech = math_methods + hallmarks + rest
    data_tech_probs = classify_group(dataset, features_data_tech, data_technologies, 'data_tech')

    features_hallmarks = math_methods + data_technologies + rest
    hallmarks_probs = classify_group(dataset, features_hallmarks, hallmarks, 'hallmarks')

    probabilities_df = pd.concat([math_probs, data_tech_probs, hallmarks_probs], axis=1)
    final_dataset = pd.concat([dataset, probabilities_df], axis=1)

    return probabilities_df, final_dataset


def classify_group_weighted_scores(dataset, math_methods, data_technologies, hallmarks, rest):
    """Per-label classification probabilities, averaged and normalized into per-group weighting vectors."""
    features_math = data_technologies + hallmarks + rest
    math_probs = classify_group_all(dataset, features_math, math_methods, 'math')
    math_probs_df = pd.DataFrame(math_probs).T.mean(axis=0) / pd.DataFrame(math_probs).T.mean(axis=0).sum()

    features_data_tech = math_methods + hallmarks + rest
    data_tech_probs = classify_group_all(dataset, features_data_tech, data_technologies, 'data_tech')
    data_tech_probs_df = pd.DataFrame(data_tech_probs).T.mean(axis=0) / pd.DataFrame(data_tech_probs).T.mean(axis=0).sum()

    features_hallmarks = math_methods + data_technologies + rest
    hallmarks_probs = classify_group_all(dataset, features_hallmarks, hallmarks, 'hallmarks')
    hallmarks_probs_df = pd.DataFrame(hallmarks_probs).T.mean(axis=0) / pd.DataFrame(hallmarks_probs).T.mean(axis=0).sum()

    return math_probs_df, data_tech_probs_df, hallmarks_probs_df


def compute_mm_dt_hm_scores(dataset, math_methods, data_technologies, hallmarks,
                             math_probs_df, data_tech_probs_df, hallmarks_probs_df):
    """Weight each group's one-hot label columns by its classification-derived weights and sum into MM/DT/Hm."""
    dataset['MM'] = dataset[math_methods].multiply(math_probs_df.values, axis=1).sum(axis=1)
    dataset['DT'] = dataset[data_technologies].multiply(data_tech_probs_df.values, axis=1).sum(axis=1)
    dataset['Hm'] = dataset[hallmarks].multiply(hallmarks_probs_df.values, axis=1).sum(axis=1)
    return dataset


def scale_mm_dt_hm_to_max(dataset):
    """Rescale MM/DT/Hm so each column's max value is 1."""
    dataset['MM'] = dataset['MM'] / dataset['MM'].max()
    dataset['DT'] = dataset['DT'] / dataset['DT'].max()
    dataset['Hm'] = dataset['Hm'] / dataset['Hm'].max()
    return dataset


# ----------------------------------------------------------------------
# 3D probability-density / trajectory plots - per cluster
# ----------------------------------------------------------------------

def assign_year_range(year, year_ranges=None):
    """Map a publication year to its '{start}-{end}' window label."""
    if year_ranges is None:
        year_ranges = YEAR_RANGES
    for start, end in year_ranges:
        if start <= year <= end:
            return f"{start}-{end}"
    return None


def calculate_cluster_centers_by_range(data, method):
    """MM/DT/Hm cluster centers, split out per year-range window."""
    cluster_range_centers = (
        data.groupby([f'{method}_Cluster', 'Year_Range'])
        .agg({'MM': 'mean', 'DT': 'mean', 'Hm': 'mean', f'{method}_Cluster': 'size'})
        .rename(columns={f'{method}_Cluster': 'Count'})
        .reset_index()
    )
    return cluster_range_centers


def create_gradient_colors(base_color, n_colors):
    rgba_color = to_rgba(base_color)
    color_list = [
        (rgba_color[0], rgba_color[1], rgba_color[2], alpha)
        for alpha in np.linspace(0.1, 1, n_colors)
    ]
    return color_list


def create_colormap(color):
    return LinearSegmentedColormap.from_list('custom_cmap', ['white', color], N=100)


def plot_2d_kde(data, axis1, axis2, grid_size=100):
    """2D KDE used by the per-cluster 3D trajectory plot (low-density cutoff at 1% of peak)."""
    kde = gaussian_kde([data[axis1], data[axis2]])
    x_min, x_max = data[axis1].min(), data[axis1].max()
    y_min, y_max = data[axis2].min(), data[axis2].max()
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde([X.ravel(), Y.ravel()]).reshape(X.shape)
    Z[Z < Z.max() * 0.01] = np.nan
    return x_min, y_min, x_max, y_max, X, Y, Z


def darken_color(color, amount=0.3):
    rgba_color = to_rgba(color)
    darkened_color = (rgba_color[0] * (1 - amount), rgba_color[1] * (1 - amount), rgba_color[2] * (1 - amount), rgba_color[3])
    return to_hex(darkened_color)


def plot_3d_with_range_centers(data, method, cluster_num, cluster_centers, range_centers, output_dir,
                                year_ranges=None, custom_colors=None):
    """3D probability-density plot for a single cluster, with per-window trajectory centers and projections."""
    if year_ranges is None:
        year_ranges = YEAR_RANGES
    if custom_colors is None:
        custom_colors = TRAJECTORY_COLORS

    cluster_data = data[data[f'{method}_Cluster'] == cluster_num]
    center_mm = cluster_centers.loc[cluster_num, 'MM']
    center_dt = cluster_centers.loc[cluster_num, 'DT']
    center_hm = cluster_centers.loc[cluster_num, 'Hm']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    cluster_color = custom_colors[cluster_num % len(custom_colors)]
    colormap = create_colormap(cluster_color)

    range_cluster_data = range_centers[range_centers[f'{method}_Cluster'] == cluster_num]
    gradient_colors = create_gradient_colors(cluster_color, len(year_ranges))

    x_min_mm_dt, y_min_mm_dt, x_max_mm_dt, y_max_mm_dt, X_mm_dt, Y_mm_dt, Z_mm_dt = plot_2d_kde(cluster_data, 'MM', 'DT')
    x_min_mm_hm, y_min_mm_hm, x_max_mm_hm, y_max_mm_hm, X_mm_hm, Y_mm_hm, Z_mm_hm = plot_2d_kde(cluster_data, 'MM', 'Hm')
    x_min_dt_hm, y_min_dt_hm, x_max_dt_hm, y_max_dt_hm, X_dt_hm, Y_dt_hm, Z_dt_hm = plot_2d_kde(cluster_data, 'DT', 'Hm')

    invert = True
    xoffset = min(x_max_mm_dt, x_max_mm_hm) if invert else 0

    ax.contourf(X_mm_dt, Y_mm_dt, Z_mm_dt, zdir='z', offset=min(y_min_mm_hm, y_min_dt_hm), cmap=colormap, alpha=0.5, zorder=1)
    ax.contourf(X_mm_hm, Z_mm_hm, Y_mm_hm, zdir='y', offset=min(y_min_mm_dt, y_min_dt_hm), cmap=colormap, alpha=0.5, zorder=2)
    ax.contourf(Z_dt_hm, X_dt_hm, Y_dt_hm, zdir='x', offset=xoffset if invert else min(x_min_mm_dt, x_min_mm_hm), cmap=colormap, alpha=0.5, zorder=3)

    prev_mm, prev_dt, prev_hm = None, None, None
    for i, (year_range, color) in enumerate(zip(year_ranges, gradient_colors)):
        range_data = range_cluster_data[range_cluster_data['Year_Range'] == f"{year_range[0]}-{year_range[1]}"]

        if not range_data.empty:
            mm, dt, hm = range_data['MM'].values[0], range_data['DT'].values[0], range_data['Hm'].values[0]

            ax.scatter(mm, dt, hm, zorder=3 + i, s=range_data['Count'].values[0] * 10,
                       color=color, edgecolor='black', alpha=0.7, label=f"{year_range[0]}-{year_range[1]}")

            ax.scatter(mm, dt, min(y_min_mm_hm, y_min_dt_hm), s=(i + 1) * 5, color=darken_color(color, amount=0.5), alpha=0.5, edgecolor='black')
            ax.scatter(mm, min(y_min_mm_dt, y_min_dt_hm), hm, s=(i + 1) * 5, color=darken_color(color, amount=0.5), alpha=0.5, edgecolor='black')
            ax.scatter(xoffset if invert else min(x_min_mm_dt, x_min_mm_hm), dt, hm, s=(i + 1) * 5, color=darken_color(color, amount=0.5), alpha=0.5, edgecolor='black')

            if prev_mm is not None:
                ax.plot([prev_mm, mm], [prev_dt, dt],       zs=[min(y_min_mm_hm, y_min_dt_hm)] * 2, linewidth=1, color=darken_color(color, amount=0.8), linestyle='--', alpha=0.5)
                ax.plot([prev_mm, mm], zs=[prev_hm, hm], ys=[min(y_min_mm_dt, y_min_dt_hm)] * 2,    linewidth=1, color=darken_color(color, amount=0.8), linestyle='--', alpha=0.5)
                ax.plot(xs=[xoffset if invert else min(x_min_mm_dt, x_min_mm_hm)] * 2, ys=[prev_dt, dt], zs=[prev_hm, hm], linewidth=1, color=darken_color(color, amount=0.8), linestyle='--', alpha=0.5)
                if invert:
                    ax.invert_xaxis()

            prev_mm, prev_dt, prev_hm = mm, dt, hm

    ax.scatter(center_mm, center_dt, center_hm, zorder=10,
               color='black', s=int(cluster_data.shape[1]) * 1, edgecolor='black', alpha=1, linewidth=1)

    ax.set_xlabel('Mathematical Methods', labelpad=8)
    ax.set_ylabel('Data-Generating Technologies', labelpad=8)
    ax.set_zlabel('Hallmarks', labelpad=6)

    ax.set_xlim([min(0.1, x_min_mm_dt), x_max_mm_dt])
    ax.set_ylim([min(0.1, x_min_dt_hm), x_max_dt_hm])
    ax.set_zlim([min(0.3, y_min_mm_hm), y_max_mm_hm])
    if invert:
        ax.invert_xaxis()

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.view_init(elev=20, azim=45)
    ax.legend(loc='upper left', framealpha=1, ncol=1, bbox_to_anchor=(0.71, 0.85), fontsize=15)

    plt.tight_layout()
    fig.subplots_adjust(left=0.3, right=1.0, top=1, bottom=0.0)

    plt.savefig(f'{output_dir}/3D_Probability_Density_of_Cluster_{cluster_num}_with_trajectories.pdf', dpi=400)
    plt.show()

    return fig


# ----------------------------------------------------------------------
# 3D probability-density / trajectory plots - overall dataset
# ----------------------------------------------------------------------

class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)


# Register the ax.arrow3D(...) convenience method used below, exactly as the
# notebook did inline.
setattr(Axes3D, 'arrow3D', _arrow3D)


def calculate_overall_centers_by_range(data):
    """MM/DT/Hm centers per year-range window, ignoring clusters."""
    overall_centers = (
        data.groupby('Year_Range')
        .agg({'MM': 'mean', 'DT': 'mean', 'Hm': 'mean', 'Year_Range': 'size'})
        .rename(columns={'Year_Range': 'Count'})
        .reset_index()
    )
    return overall_centers


def calculate_2d_kde(data, axis1, axis2, grid_size=100):
    """2D KDE used by the overall-dataset 3D trajectory plot (low-density cutoff at 10% of peak)."""
    kde = gaussian_kde([data[axis1], data[axis2]])
    x_min, x_max = data[axis1].min(), data[axis1].max()
    y_min, y_max = data[axis2].min(), data[axis2].max()
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde([X.ravel(), Y.ravel()]).reshape(X.shape)
    Z[Z < Z.max() * 0.1] = np.nan
    return x_min, x_max, y_min, y_max, X, Y, Z


def plot_overall_centers_with_projections(data, overall_centers, output_dir, include_AI,
                                           filename_prefix='3D_Probability_Density_of_full_dataset_with_trajectories',
                                           year_ranges=None, custom_colors=None):
    """3D contour + trajectory plot of the overall (non-cluster) MM/DT/Hm centers per year-range window.

    Used for both the probability-weighted scores (filename_prefix=
    '3D_Probability_Density_of_full_dataset_with_trajectories') and the
    max-scaled scores (filename_prefix='3D_Summation_of_full_dataset_with_trajectories').
    """
    if year_ranges is None:
        year_ranges = YEAR_RANGES
    if custom_colors is None:
        custom_colors = TRAJECTORY_COLORS

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x_min_mm_dt, x_max_mm_dt, y_min_mm_dt, y_max_mm_dt, X_mm_dt, Y_mm_dt, Z_mm_dt = calculate_2d_kde(data, 'MM', 'DT')
    x_min_mm_hm, x_max_mm_hm, y_min_mm_hm, y_max_mm_hm, X_mm_hm, Y_mm_hm, Z_mm_hm = calculate_2d_kde(data, 'MM', 'Hm')
    x_min_dt_hm, x_max_dt_hm, y_min_dt_hm, y_max_dt_hm, X_dt_hm, Y_dt_hm, Z_dt_hm = calculate_2d_kde(data, 'DT', 'Hm')

    invert = True
    xoffset = 1 if invert else 0

    ax.contourf(X_mm_dt, Y_mm_dt, Z_mm_dt, zdir='z', offset=y_min_mm_hm, cmap='Oranges', alpha=0.4)
    ax.contourf(X_mm_hm, Z_mm_hm, Y_mm_hm, zdir='y', offset=y_min_mm_dt, cmap='Greens', alpha=0.4)
    ax.contourf(Z_dt_hm, X_dt_hm, Y_dt_hm, zdir='x', offset=xoffset, cmap='Blues', alpha=0.4)

    prev_mm, prev_dt, prev_hm = None, None, None
    for i, year_range in enumerate(year_ranges):
        range_label = f"{year_range[0]}-{year_range[1]}"
        range_data = overall_centers[overall_centers['Year_Range'] == range_label]

        if not range_data.empty:
            mm, dt, hm = range_data['MM'].values[0], range_data['DT'].values[0], range_data['Hm'].values[0]
            size = range_data['Count'].values[0] * 5

            ax.scatter(mm, dt, hm, s=size, color='black', edgecolor='black', alpha=0.7, label=f"{range_label}")

            ax.scatter(mm, dt, y_min_mm_hm, s=(i + 1) * 20, color=darken_color('orange'), alpha=0.5, edgecolor='black')
            ax.scatter(mm, y_min_mm_dt, hm, s=(i + 1) * 20, color=darken_color('green'), alpha=0.5, edgecolor='black')
            ax.scatter(xoffset, dt, hm, s=(i + 1) * 20, color=darken_color('blue'), alpha=0.5, edgecolor='black')

            if prev_mm is not None:
                ax.plot([prev_mm, mm], [prev_dt, dt], zs=[y_min_mm_hm] * 2, color=darken_color('orange'), linestyle='--', alpha=0.5)
                ax.plot([prev_mm, mm], zs=[prev_hm, hm], ys=[y_min_mm_dt] * 2, color=darken_color('green'), linestyle='--', alpha=0.5)
                ax.plot(xs=[xoffset] * 2, ys=[prev_dt, dt], zs=[prev_hm, hm], color=darken_color('blue'), linestyle='--', alpha=0.5)

            prev_mm, prev_dt, prev_hm = mm, dt, hm

    last_two = overall_centers.iloc[-2:]
    count_diff = last_two['Count'].values[1] - last_two['Count'].values[0]
    mm_diff = last_two['MM'].values[1] - last_two['MM'].values[0]
    dt_diff = last_two['DT'].values[1] - last_two['DT'].values[0]
    hm_diff = last_two['Hm'].values[1] - last_two['Hm'].values[0]
    scaling_factor = 0.1

    p_vector = scaling_factor * np.abs(count_diff) * np.array([mm_diff, dt_diff, hm_diff])

    last_mm, last_dt, last_hm = last_two['MM'].values[1], last_two['DT'].values[1], last_two['Hm'].values[1]

    ax.arrow3D(last_mm, last_dt, last_hm, p_vector[0], p_vector[1], p_vector[2], mutation_scale=20, arrowstyle="-|>", color='black', linestyle='solid')
    ax.arrow3D(xoffset, last_dt, last_hm, 0, p_vector[1], p_vector[2], mutation_scale=20, arrowstyle="-|>", color='blue', linestyle='solid')
    ax.arrow3D(last_mm, 0, last_hm, p_vector[0], 0, p_vector[2], mutation_scale=20, arrowstyle="-|>", color='darkgreen', linestyle='solid')
    ax.arrow3D(last_mm, last_dt, 0, p_vector[0], p_vector[1], 0, mutation_scale=20, arrowstyle="-|>", color='brown', linestyle='solid')

    ax.set_xlabel('Mathematical Methods', labelpad=8)
    ax.set_ylabel('Data-Generating Technologies', labelpad=8)
    ax.set_zlabel('Hallmarks', labelpad=6)
    ax.set_xlim([x_min_mm_dt, xoffset])
    ax.set_ylim([y_min_mm_dt, xoffset])
    ax.set_zlim([y_min_mm_hm, xoffset])
    if invert:
        ax.invert_xaxis()

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.view_init(elev=20, azim=45)
    ax.legend(loc='upper left', framealpha=1, ncol=1, bbox_to_anchor=(0.71, 0.76), fontsize=15)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{filename_prefix}.pdf' if include_AI else
                f'{output_dir}/{filename_prefix}_no_AI.pdf', dpi=400)

    plt.show()

    return fig


# ----------------------------------------------------------------------
# Figure post-processing
# ----------------------------------------------------------------------

def crop_output_pdfs(output_dir, include_AI, cluster_count):
    """Crop the overall probability/summation figures and every per-cluster figure with `pdfcrop`."""
    probability_file = "3D_Probability_Density_of_full_dataset_with_trajectories.pdf" if include_AI else "3D_Probability_Density_of_full_dataset_with_trajectories_no_AI.pdf"
    summation_file = "3D_Summation_of_full_dataset_with_trajectories.pdf" if include_AI else "3D_Summation_of_full_dataset_with_trajectories_no_AI.pdf"

    if os.path.exists(os.path.join(output_dir, probability_file)):
        print(f"Cropping {probability_file} in {output_dir}...")
        subprocess.run(["pdfcrop", probability_file, probability_file], cwd=output_dir)

    if os.path.exists(os.path.join(output_dir, summation_file)):
        print(f"Cropping {summation_file} in {output_dir}...")
        subprocess.run(["pdfcrop", summation_file, summation_file], cwd=output_dir)

    for c in reversed(range(1, cluster_count + 1)):
        file_name = f"3D_Probability_Density_of_Cluster_{c}_with_trajectories.pdf"
        subprocess.run(["pdfcrop", file_name, file_name], cwd=output_dir)
        if os.path.exists(os.path.join(output_dir, file_name)):
            print(f"Cropping {file_name}...")
        else:
            print(f"Warning: {file_name} not found in {output_dir}")
