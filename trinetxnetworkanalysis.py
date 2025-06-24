
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
st.title("📊 TriNetX Collaboration Dashboard: Participation & Equity Insights")

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

    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df.head())

    # Time filter
    start_date = st.date_input("Start Date", df['Date'].min().date())
    end_date = st.date_input("End Date", df['Date'].max().date())
    slice_freq = st.selectbox("Select Time Slice", ["M", "Q", "6M", "A"], index=0, format_func=lambda x: {
        "M": "Monthly", "Q": "Quarterly", "6M": "Semiannually", "A": "Annually"
    }[x])
    df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

    st.sidebar.subheader("Filters")
    urim_only = st.sidebar.checkbox("URIM Only")
    firstgen_only = st.sidebar.checkbox("FirstGen Only")
    enable_clustering = st.sidebar.checkbox("Auto-Cluster Users (Community Detection)", value=True)

    if urim_only:
        df = df[df['URIM'] == True]
    if firstgen_only:
        df = df[df['FirstGen'] == True]

    if 'Project_ID' in df.columns:
        project_filter = st.sidebar.multiselect("Filter by Project ID", sorted(df['Project_ID'].dropna().unique()))
        if project_filter:
            df = df[df['Project_ID'].isin(project_filter)]
    else:
        st.warning("⚠️ 'Project_ID' column not found in the data.")
        st.stop()

    # NETWORK CONSTRUCTION
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

    # Degree and centrality
    degree_dict = dict(user_graph.degree())
    centrality = nx.betweenness_centrality(user_graph)
    df_metrics = pd.DataFrame({
        "User_ID": list(degree_dict.keys()),
        "Degree": list(degree_dict.values()),
        "Betweenness Centrality": [centrality[u] for u in degree_dict.keys()],
        "Cluster": [cluster_map.get(u, -1) for u in degree_dict.keys()]
    })
    merged_df = pd.merge(df_metrics, df[['User_ID', 'GroupedRole', 'Role', 'URIM', 'FirstGen']].drop_duplicates(), on="User_ID", how="left")

    with st.expander("📈 Time-Sliced Participation Summary", expanded=True):
        df_time = df.groupby(pd.Grouper(key="Date", freq=slice_freq)).agg(
            Total_Students=('User_ID', lambda x: df[df['GroupedRole'] == 'Student']['User_ID'].nunique()),
            Total_Projects=('Project_ID', pd.Series.nunique)
        ).reset_index()
        df_time['Pct_Projects_With_Students'] = [
            df[(df['Date'].dt.to_period(slice_freq) == period) &
               (df['GroupedRole'] == 'Student')]['Project_ID'].nunique() / total if total else 0
            for period, total in zip(df_time['Date'].dt.to_period(slice_freq), df_time['Total_Projects'])
        ]
        st.dataframe(df_time)

    with st.expander("📊 Equity Watch: URIM & FirstGen Participation", expanded=False):
        eq_stats = df.groupby([pd.Grouper(key="Date", freq=slice_freq), 'GroupedRole']).agg(
            Total=('User_ID', 'nunique'),
            URIM_Count=('URIM', 'sum'),
            FirstGen_Count=('FirstGen', 'sum')
        ).reset_index()
        eq_stats['URIM_%'] = eq_stats['URIM_Count'] / eq_stats['Total']
        eq_stats['FirstGen_%'] = eq_stats['FirstGen_Count'] / eq_stats['Total']
        st.dataframe(eq_stats)

    with st.expander("📈 Student Entry & Duration Tracking", expanded=False):
        first_appearance = df[df['GroupedRole'] == 'Student'].groupby('User_ID')['Date'].min().reset_index()
        last_appearance = df[df['GroupedRole'] == 'Student'].groupby('User_ID')['Date'].max().reset_index()
        student_life = pd.merge(first_appearance, last_appearance, on='User_ID', suffixes=('_Start', '_End'))
        student_life['Duration_Days'] = (student_life['Date_End'] - student_life['Date_Start']).dt.days
        st.dataframe(student_life)

    with st.expander("📈 Centrality Over Time by Grouped Role", expanded=True):
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

        df_time_central = pd.DataFrame(time_results)
        grouped = df_time_central.groupby(['Date', 'GroupedRole']).agg({
            'Degree': 'mean',
            'Centrality': 'mean'
        }).reset_index()

        fig, ax = plt.subplots()
        for role in grouped['GroupedRole'].unique():
            role_data = grouped[grouped['GroupedRole'] == role]
            ax.plot(role_data['Date'], role_data['Centrality'], marker='o', label=f"{role} Centrality")
        ax.set_ylabel("Avg Betweenness Centrality")
        ax.set_title("Betweenness Centrality Over Time")
        ax.legend()
        st.pyplot(fig)

    with st.expander("🌐 Interactive Network (Clustered)", expanded=False):
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
