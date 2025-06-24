
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
from networkx.algorithms.community import greedy_modularity_communities
from scipy.stats import chi2_contingency, ttest_ind

st.set_page_config(page_title="TriNetX Research Access Dashboard", layout="wide")
st.title("📊 TriNetX Student Research Participation Dashboard")

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

    # Sidebar filters
    st.sidebar.title("📌 Filters & Options")
    start_date = st.sidebar.date_input("Start Date", df['Date'].min().date())
    end_date = st.sidebar.date_input("End Date", df['Date'].max().date())
    slice_freq = st.sidebar.selectbox("Time Slice", ["M", "Q", "6M", "A"], format_func=lambda x: {
        "M": "Monthly", "Q": "Quarterly", "6M": "Semiannually", "A": "Annually"
    }[x])
    df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

    show_participation = st.sidebar.checkbox("Show Time-Based Participation", value=True)
    show_equity_stats = st.sidebar.checkbox("Show URIM/FirstGen Equity Watch", value=True)
    show_network = st.sidebar.checkbox("Show Network Visualization", value=False)
    show_chi2 = st.sidebar.checkbox("Run Chi-Square Test", value=True)
    show_ttest = st.sidebar.checkbox("Run T-Test on Centrality", value=True)

    if show_participation:
        st.subheader("📈 Time-Based Participation Summary")
        time_summary = df.groupby([pd.Grouper(key="Date", freq=slice_freq), 'GroupedRole'])['User_ID'].nunique().reset_index()
        pivot = time_summary.pivot(index='Date', columns='GroupedRole', values='User_ID').fillna(0)
        st.line_chart(pivot)

    if show_equity_stats:
        st.subheader("📊 URIM and FirstGen Participation Over Time")
        eq_stats = df.groupby([pd.Grouper(key="Date", freq=slice_freq), 'GroupedRole']).agg(
            Total=('User_ID', 'nunique'),
            URIM_Count=('URIM', 'sum'),
            FirstGen_Count=('FirstGen', 'sum')
        ).reset_index()

        eq_stats['Total'] = pd.to_numeric(eq_stats['Total'], errors='coerce').fillna(0)
        eq_stats['URIM_Count'] = pd.to_numeric(eq_stats['URIM_Count'], errors='coerce').fillna(0)
        eq_stats['FirstGen_Count'] = pd.to_numeric(eq_stats['FirstGen_Count'], errors='coerce').fillna(0)

        eq_stats['URIM_%'] = eq_stats.apply(
            lambda row: row['URIM_Count'] / row['Total'] if row['Total'] > 0 else 0, axis=1)
        eq_stats['FirstGen_%'] = eq_stats.apply(
            lambda row: row['FirstGen_Count'] / row['Total'] if row['Total'] > 0 else 0, axis=1)

        for metric in ['URIM_%', 'FirstGen_%']:
            fig, ax = plt.subplots()
            for role in eq_stats['GroupedRole'].unique():
                subset = eq_stats[eq_stats['GroupedRole'] == role]
                ax.plot(subset['Date'], subset[metric], marker='o', label=role)
            ax.set_title(metric + " by Role")
            ax.set_ylabel(metric)
            ax.legend()
            st.pyplot(fig)

    if show_chi2:
        st.subheader("🧪 Chi-Square Test: URIM Representation by Role")
        urim_table = df[df['GroupedRole'].isin(['Mentor', 'Student'])]
        contingency = pd.crosstab(urim_table['GroupedRole'], urim_table['URIM'])
        if contingency.shape == (2, 2):
            chi2, p_val, dof, _ = chi2_contingency(contingency)
            st.write("Chi² =", chi2, "| p =", p_val)
            st.dataframe(contingency)
        else:
            st.warning("Insufficient data for chi-square test.")

    if show_ttest:
        st.subheader("📐 T-Test: Centrality Comparison Between Mentors and Students")
        df['Centrality'] = df.groupby('User_ID')['Project_ID'].transform('count')
        student_central = df[df['GroupedRole'] == 'Student']['Centrality']
        mentor_central = df[df['GroupedRole'] == 'Mentor']['Centrality']
        if len(student_central) > 1 and len(mentor_central) > 1:
            t_stat, t_p = ttest_ind(student_central, mentor_central, equal_var=False)
            st.write("T-statistic =", t_stat, "| p =", t_p)
        else:
            st.warning("Not enough data for t-test.")

    if show_network:
        st.subheader("🌐 Network Visualization")
        B = nx.Graph()
        users = df['User_ID'].unique()
        projects = df['Project_ID'].unique()
        B.add_nodes_from(users, bipartite=0)
        B.add_nodes_from(projects, bipartite=1)
        for _, row in df.iterrows():
            B.add_edge(row['User_ID'], row['Project_ID'])

        G = nx.projected_graph(B, users)
        degree_dict = dict(G.degree())
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
        for node in G.nodes():
            label = f"{node} | Degree: {degree_dict.get(node,0)}"
            net.add_node(node, label=label)
        for u, v in G.edges():
            net.add_edge(u, v)

        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.html') as tmp:
            net.save_graph(tmp.name)
            components.html(tmp.read(), height=600, scrolling=True)

else:
    st.info("👈 Upload a TriNetX usage log file to get started.")
