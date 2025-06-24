
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="TriNetX Collaboration Dashboard", layout="wide")

st.title("📊 TriNetX Social Network Collaboration Dashboard")

uploaded_file = st.file_uploader("Upload TriNetX Usage Log CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    with st.expander("🔍 Preview & Filter Data"):
        st.dataframe(df.head())
        start_date = st.date_input("Start Date", df['Date'].min().date())
        end_date = st.date_input("End Date", df['Date'].max().date())
        role_filter = st.multiselect("Filter by Role", df['Role'].unique(), default=list(df['Role'].unique()))
        df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
        df = df[df['Role'].isin(role_filter)]

    st.sidebar.subheader("Subnetwork Filters")
    urim_only = st.sidebar.checkbox("URIM Only")
    firstgen_only = st.sidebar.checkbox("FirstGen Only")
    project_filter = st.sidebar.multiselect("Filter by Project ID", sorted(df['Project ID'].unique()))

    if urim_only:
        df = df[df['URIM'] == True]
    if firstgen_only:
        df = df[df['FirstGen'] == True]
    if project_filter:
        df = df[df['Project ID'].isin(project_filter)]

    B = nx.Graph()
    user_nodes = df['User ID'].unique()
    project_nodes = df['Project ID'].unique()
    B.add_nodes_from(user_nodes, bipartite=0, type='user')
    B.add_nodes_from(project_nodes, bipartite=1, type='project')

    for _, row in df.iterrows():
        B.add_edge(row['User ID'], row['Project ID'])

    user_graph = nx.projected_graph(B, user_nodes)

    degree_dict = dict(user_graph.degree())
    centrality = nx.betweenness_centrality(user_graph)
    df_metrics = pd.DataFrame({
        "User ID": list(degree_dict.keys()),
        "Degree": list(degree_dict.values()),
        "Betweenness Centrality": [centrality[u] for u in degree_dict.keys()]
    })
    merged_df = pd.merge(df_metrics, df[['User ID', 'Role', 'URIM', 'FirstGen']].drop_duplicates(), on="User ID", how="left")

    st.subheader("📋 Collaboration Metrics Table")
    st.dataframe(merged_df)
    st.download_button("📥 Download Metrics Table", data=merged_df.to_csv(index=False), file_name="metrics.csv")

    st.subheader("📊 Co-authorship Heatmap")
    coauthorship = pd.crosstab(df['User ID'], df['Project ID'])
    co_matrix = coauthorship.dot(coauthorship.T)
    co_matrix[co_matrix < 2] = 0
    fig_hm, ax_hm = plt.subplots(figsize=(10, 8))
    sns.heatmap(co_matrix, cmap="Blues", ax=ax_hm)
    ax_hm.set_title("Co-authorship Heatmap")
    st.pyplot(fig_hm)

    st.subheader("📈 User Centrality Over Time")
    centrality_times = []
    for date in pd.date_range(df['Date'].min(), df['Date'].max(), freq='M'):
        temp_df = df[df['Date'] <= date]
        temp_B = nx.Graph()
        temp_B.add_nodes_from(temp_df['User ID'].unique())
        for _, row in temp_df.iterrows():
            temp_B.add_edge(row['User ID'], row['Project ID'])
        temp_graph = nx.projected_graph(temp_B, temp_df['User ID'].unique())
        central = nx.betweenness_centrality(temp_graph)
        for user, cent in central.items():
            centrality_times.append({'User ID': user, 'Date': date, 'Centrality': cent})
    centrality_df = pd.DataFrame(centrality_times)

    selected_user_ct = st.selectbox("Select a User ID for longitudinal centrality view", sorted(df['User ID'].unique()))
    user_centrality_plot = centrality_df[centrality_df['User ID'] == selected_user_ct]
    fig_ct, ax_ct = plt.subplots()
    ax_ct.plot(user_centrality_plot['Date'], user_centrality_plot['Centrality'], marker='o')
    ax_ct.set_title(f"Betweenness Centrality Over Time: {selected_user_ct}")
    ax_ct.set_ylabel("Centrality")
    ax_ct.set_xlabel("Date")
    st.pyplot(fig_ct)

    st.subheader("🕵️ Explore Ego Network of a User")
    selected_user = st.selectbox("Select a User ID to visualize ego network", options=sorted(user_nodes))
    ego_net = nx.ego_graph(user_graph, selected_user)
    ego_viz = Network(height="500px", width="100%", notebook=False, bgcolor="#f0f0f0", font_color="black")
    ego_viz.from_nx(ego_net)

    for node in ego_viz.nodes:
        if node['id'] == selected_user:
            node['color'] = '#ff0000'
            node['title'] = f"{node['id']} (Selected User)"
        else:
            node['color'] = '#87cefa'

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.html') as tmp_file:
        ego_viz.save_graph(tmp_file.name)
        components.html(tmp_file.read(), height=500, scrolling=True)

    st.subheader("📅 Project Timeline")
    proj_df = df.groupby(df['Project ID'])['Date'].min().reset_index()
    proj_df = proj_df.sort_values("Date")
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    ax5.scatter(proj_df['Date'], proj_df['Project ID'], alpha=0.6)
    ax5.set_xlabel("Start Date")
    ax5.set_ylabel("Project ID")
    ax5.set_title("Timeline of Project Initiations")
    st.pyplot(fig5)

else:
    st.info("👈 Upload a TriNetX usage log file to get started.")
