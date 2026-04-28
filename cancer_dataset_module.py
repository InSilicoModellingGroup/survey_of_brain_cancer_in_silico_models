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
 