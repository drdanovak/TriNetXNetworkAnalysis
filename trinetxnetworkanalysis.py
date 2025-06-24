
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
st.title("📊 TriNetX Collaboration Dashboard (Grouped Roles + Time Slices)")

uploaded_file = st.file_uploader("Upload TriNetX Usage Log CSV", type=["csv"])

role_map = {
    "Clinical GSR": "Mentor",
    "Physician - Community/Voluntary Faculty": "Mentor",
    "Physician - UCR Health Employee": "Mentor",
    "UCR Academic Faculty (SMPPH, Biomedicine, Statistics, or other)": "Mentor",
    "UCR Resident": "Mentor",
    "Graduate Student (SMPPH, Statistics, Data Science, Etc.)": "Student",
    "Medical Student": "Student",
    "Undergraduate Student": "Student"
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    df['GroupedRole'] = df['Role'].map(role_map).fillna("Other")

    st.write("Available columns:", df.columns.tolist())

    with st.expander("🔍 Preview & Filter Data"):
        st.dataframe(df.head())
        start_date = st.date_input("Start Date", df['Date'].min().date())
        end_date = st.date_input("End Date", df['Date'].max().date())
        slice_freq = st.selectbox("Select Time Slice", ["M", "Q", "6M", "A"], index=0, format_func=lambda x: {
            "M": "Monthly", "Q": "Quarterly", "6M": "Semiannually", "A": "Annually"
        }[x])
        df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

    st.sidebar.subheader("Subnetwork Filters")
    urim_only = st.sidebar.checkbox("URIM Only")
    firstgen_only = st.sidebar.checkbox("FirstGen Only")
    enable_clustering = st.sidebar.checkbox("Auto-Cluster Users (Community Detection)", value=True)

    if 'Project_ID' in df.columns:
        project_filter = st.sidebar.multiselect("Filter by Project ID", sorted(df['Project_ID'].dropna().unique()))
        if project_filter:
            df = df[df['Project_ID'].isin(project_filter)]
    else:
        st.warning("⚠️ 'Project_ID' column not found in the data.")
        st.stop()

    if urim_only:
        df = df[df['URIM'] == True]
    if firstgen_only:
        df = df[df['FirstGen'] == True]

    B = nx.Graph()
    user_nodes = df['User_ID'].unique()
    project_nodes = df['Project_ID'].unique()
    B.add_nodes_from(user_nodes, bipartite=0, type='user')
    B.add_nodes_from(project_nodes, bipartite=1, type='project')

    for _, row in df.iterrows():
        B.add_edge(row['User_ID'], row['Project_ID'])

    user_graph = nx.projected_graph(B, user_nodes)

    cluster_map = {}
    if enable_clustering:
        communities = list(greedy_modularity_communities(user_graph))
        for i, group in enumerate(communities):
            for user in group:
                cluster_map[user] = i

    degree_dict = dict(user_graph.degree())
    centrality = nx.betweenness_centrality(user_graph)
    df_metrics = pd.DataFrame({
        "User_ID": list(degree_dict.keys()),
        "Degree": list(degree_dict.values()),
        "Betweenness Centrality": [centrality[u] for u in degree_dict.keys()],
        "Cluster": [cluster_map.get(u, -1) for u in degree_dict.keys()]
    })
    merged_df = pd.merge(df_metrics, df[['User_ID', 'GroupedRole', 'Role', 'URIM', 'FirstGen']].drop_duplicates(), on="User_ID", how="left")

    st.subheader("📋 Collaboration Metrics Table")
    st.dataframe(merged_df)
    st.download_button("📥 Download Metrics Table", data=merged_df.to_csv(index=False), file_name="metrics.csv")

    st.subheader("📈 Centrality Over Time by Grouped Role")
    time_results = []
    for time, gdf in df.groupby(pd.Grouper(key="Date", freq=slice_freq)):
        B_temp = nx.Graph()
        B_temp.add_nodes_from(gdf['User_ID'].unique())
        for _, row in gdf.iterrows():
            B_temp.add_edge(row['User_ID'], row['Project_ID'])
        G_temp = nx.projected_graph(B_temp, gdf['User_ID'].unique())
        if len(G_temp.nodes) > 0:
            temp_centrality = nx.betweenness_centrality(G_temp)
            temp_degree = dict(G_temp.degree())
            for user in G_temp.nodes:
                role = df[df['User_ID'] == user]['GroupedRole'].values[0]
                time_results.append({
                    "Date": time,
                    "User_ID": user,
                    "GroupedRole": role,
                    "Degree": temp_degree[user],
                    "Centrality": temp_centrality[user]
                })

    df_time = pd.DataFrame(time_results)
    grouped = df_time.groupby(['Date', 'GroupedRole']).agg({
        'Degree': 'mean',
        'Centrality': 'mean'
    }).reset_index()

    fig, ax = plt.subplots()
    for role in grouped['GroupedRole'].unique():
        role_data = grouped[grouped['GroupedRole'] == role]
        ax.plot(role_data['Date'], role_data['Centrality'], marker='o', label=f"{role} Centrality")
    ax.set_ylabel("Average Betweenness Centrality")
    ax.set_title("Betweenness Centrality Over Time by Role Group")
    ax.legend()
    st.pyplot(fig)

    st.subheader("🌐 Interactive Network (Colored by Cluster)")
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
