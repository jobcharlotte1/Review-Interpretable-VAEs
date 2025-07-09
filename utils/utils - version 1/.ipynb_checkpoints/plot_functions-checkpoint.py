import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # For color palette


def plot_df_melted(df_melted, value_name, x_value, y_value, column_order, color_order, color_value, colors_list, title_name):

    fig = px.bar(df_melted.sort_values(by=value_name, ascending=False), x=x_value, y=y_value, category_orders={column_order: df_melted[column_order].tolist(), color_order: df_melted[color_order].tolist()},
                 color=color_value, color_discrete_sequence=colors_list, barmode="group", title=title_name)

    # Rotate x-axis labels
    fig.update_layout(xaxis_tickangle=-45)

    # Show the plot
    fig.show()
    
    
def create_umap_scatter(umap_coords, labels, title):
    # Convert categorical labels to numerical values
    labels_cat = pd.Categorical(labels)
    labels_numeric = labels_cat.codes  # Convert categories to integer codes
    
    # Generate a color scale for the clusters dynamically
    num_clusters = len(labels_cat.categories)
    
    # Choose a color palette from Plotly, cycling through if necessary
    colors = px.colors.qualitative.Set1 * ((num_clusters // len(px.colors.qualitative.Set1)) + 1)  # Cycle the colors if there are more than 9 clusters
    colors = colors[:num_clusters]  # Limit the colors to the number of clusters
    
    # Create a scatter plot with discrete color assignments
    scatter = go.Scatter(
        x=umap_coords[:, 0], 
        y=umap_coords[:, 1], 
        mode='markers', 
        marker=dict(
            color=[colors[labels_cat.codes[i]] for i in range(len(labels))],  # Map labels to colors
            size=6,  # Adjust size of the points
            showscale=False,  # No global color bar
        ),
        text=labels,
        name=title,
        showlegend=True  # Enable legend for discrete colors
    )
    
    return scatter, labels_cat.categories, colors  # Return the cluster categories and corresponding colors
