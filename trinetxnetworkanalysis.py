import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns

from networkx.algorithms.community import greedy_modularity_communities

st.set_page_config(page_title="TriNetX Collaboration Dashboard", layout="wide")

st.title("📊 TriNetX Social Network Collaboration Dashboard")

uploaded_file = st.file_uploader("Upload TriNetX Usage Log CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])

    st.write("Available columns:", df.columns.tolist())

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
    instance_filter = st.sidebar.multiselect("Filter by Instance", sorted(df['Instance'].unique()))
    enable_clustering = st.sidebar.checkbox("Auto-Cluster Users (Community Detection)", value=True)

    if urim_only:
        df = df[df['URIM'] == True]
    if firstgen_only:
        df = df[df['FirstGen'] == True]
    if instance_filter:
        df = df[df['Instance'].isin(instance_filter)]

    B = nx.Graph()
    user_nodes = df['User'].unique()
    instance_nodes = df['Instance'].unique()
    B.add_nodes_from(user_nodes, bipartite=0, type='user')
    B.add_nodes_from(instance_nodes, bipartite=1, type='instance')

    for _, row in df.iterrows():
        B.add_edge(row['User'], row['Instance'])

    user_graph = nx.projected_graph(B, user_nodes)

    # Community detection
    cluster_map = {}
    if enable_clustering:
        communities = list(greedy_modularity_communities(user_graph))
        for i, group in enumerate(communities):
            for user in group:
                cluster_map[user] = i

    # Build metrics
    degree_dict = dict(user_graph.degree())
    centrality = nx.betweenness_centrality(user_graph)
    df_metrics = pd.DataFrame({
        "User": list(degree_dict.keys()),
        "Degree": list(degree_dict.values()),
        "Betweenness Centrality": [centrality[u] for u in degree_dict.keys()],
        "Cluster": [cluster_map.get(u, -1) for u in degree_dict.keys()]
    })
    merged_df = pd.merge(df_metrics, df[['User', 'Role', 'URIM', 'FirstGen']].drop_duplicates(), on="User", how="left")

    st.subheader("📋 Collaboration Metrics Table")
    st.dataframe(merged_df)
    st.download_button("📥 Download Metrics Table", data=merged_df.to_csv(index=False), file_name="metrics.csv")

    st.subheader("🌐 Interactive User Collaboration Network")
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    for node in user_graph.nodes():
        label = f"{node} | Degree: {degree_dict[node]}"
        title = f"User: {node}\nDegree: {degree_dict[node]}\nCentrality: {centrality[node]:.3f}"
        color = f"hsl({(cluster_map.get(node, 0) * 37) % 360}, 70%, 70%)" if enable_clustering else "#97c2fc"
        net.add_node(node, label=label, title=title, color=color)
    for u, v in user_graph.edges():
        net.add_edge(u, v)

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        components.html(tmp_file.read(), height=600, scrolling=True)

else:
    st.info("👈 Upload a TriNetX usage log file to get started.")
