
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


st.set_page_config(page_title="TriNetX Collaboration Dashboard", layout="wide")
st.title("📊 TriNetX Collaboration Dashboard: Equity + Statistical Testing")

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

    start_date = st.date_input("Start Date", df['Date'].min().date())
    end_date = st.date_input("End Date", df['Date'].max().date())
    slice_freq = st.selectbox("Time Slice", ["M", "Q", "6M", "A"], index=1, format_func=lambda x: {
        "M": "Monthly", "Q": "Quarterly", "6M": "Semiannually", "A": "Annually"
    }[x])
    df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]

    # Equity Watch Panel with statistical tests
    with st.expander("📊 Equity Watch with Statistical Tests", expanded=True):
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

        st.dataframe(eq_stats)

        # Chi-square test for URIM participation by role
        urim_table = df[df['GroupedRole'].isin(['Mentor', 'Student'])]
        contingency = pd.crosstab(urim_table['GroupedRole'], urim_table['URIM'])
        chi2, p_val, dof, expected = chi2_contingency(contingency)

        st.markdown("#### Chi-Square Test: URIM Representation by Role")
        st.write("Chi2:", chi2, "| p-value:", p_val)
        st.write("Contingency Table:")
        st.dataframe(contingency)

        # T-test for centrality difference (if later calculated)
        st.markdown("#### T-Test: Centrality Comparison Between Mentors and Students")
        centralities = df.copy()
        centralities['Centrality'] = centralities.groupby('User_ID')['Project_ID'].transform('count')
        student_central = centralities[centralities['GroupedRole'] == 'Student']['Centrality']
        mentor_central = centralities[centralities['GroupedRole'] == 'Mentor']['Centrality']

        if len(student_central) > 1 and len(mentor_central) > 1:
            t_stat, t_p_val = ttest_ind(student_central, mentor_central, equal_var=False)
            st.write("T-Statistic:", t_stat, "| p-value:", t_p_val)
        else:
            st.warning("Not enough data to run t-test.")
else:
    st.info("👈 Upload a TriNetX usage log file to get started.")
