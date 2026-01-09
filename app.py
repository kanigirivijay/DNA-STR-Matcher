    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from collections import defaultdict
    import networkx as nx
    from io import BytesIO
    import base64
    import time

    # --- Ensure session state keys exist ---
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    if 'trained_models' not in st.session_state:
        st.session_state['trained_models'] = {}
    if 'individuals' not in st.session_state:
        st.session_state['individuals'] = {}
    if 'match_df' not in st.session_state:
        st.session_state['match_df'] = pd.DataFrame()
    if 'partial_df' not in st.session_state:
        st.session_state['partial_df'] = pd.DataFrame()
    if 'allele_df' not in st.session_state:
        st.session_state['allele_df'] = pd.DataFrame()

    # --- Movie-Style Cinematic UI/UX ---
    st.set_page_config(page_title="🧬 DNA STR Matcher: Movie Mode", layout="wide", page_icon="🧬")

    # Movie-style CSS and fonts
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
    <style>
    body {
        background: radial-gradient(ellipse at center, #0f2027 0%, #2c5364 100%);
        background-attachment: fixed;
    }
    .stApp {
        background: rgba(10, 20, 40, 0.98);
        border-radius: 20px;
        color: #fff;
    }
    h1, h2, h3, h4 {
        font-family: 'Orbitron', 'Share Tech Mono', monospace;
        color: #00fff7;
        text-shadow: 0 0 20px #00fff7, 0 0 40px #00fff7;
        letter-spacing: 2px;
    }
    .glass-card {
        background: rgba(30, 60, 90, 0.7);
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 #00fff799;
        backdrop-filter: blur(8px);
        border: 1px solid #00fff7;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        animation: pulseGlow 2s infinite alternate;
    }
    @keyframes pulseGlow {
        from { box-shadow: 0 0 20px #00fff7; }
        to   { box-shadow: 0 0 40px #00fff7, 0 0 80px #00fff7; }
    }
    .stButton>button {
        background: linear-gradient(90deg, #00fff7 0%, #00ff99 100%);
        color: #000;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 0 20px #00fff7;
        transition: 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00ff99 0%, #00fff7 100%);
        box-shadow: 0 0 40px #00fff7;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: bold;
        color: #00fff7;
    }
    ::-webkit-scrollbar-thumb {
        background: #00fff7;
        border-radius: 8px;
    }
    ::-webkit-scrollbar {
        width: 8px;
        background: #1a1a40;
    }
    </style>
    """, unsafe_allow_html=True)

    # Animated DNA SVG header
    st.markdown("""
    <div style="text-align:center;">
    <svg width="180" height="120" viewBox="0 0 180 120">
    <g>
        <ellipse cx="90" cy="60" rx="70" ry="50" fill="none" stroke="#00fff7" stroke-width="4"/>
        <ellipse cx="90" cy="60" rx="70" ry="50" fill="none" stroke="#00ff99" stroke-width="4" transform="rotate(30 90 60)"/>
        <ellipse cx="90" cy="60" rx="70" ry="50" fill="none" stroke="#00fff7" stroke-width="4" transform="rotate(-30 90 60)"/>
        <animateTransform attributeName="transform" attributeType="XML" type="rotate" from="0 90 60" to="360 90 60" dur="8s" repeatCount="indefinite"/>
    </g>
    </svg>
    <h1>DNA STR Matcher Ultra</h1>
    <div style='font-size:1.2rem; color:#00fff7; text-shadow:0 0 10px #00fff7;'>Next-Gen Forensic DNA Analysis Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Neon-glow instructions and progress
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;'>
            <span style='font-size:60px; text-shadow:0 0 20px #00fff7;'>🧬</span>
            <h2 style='color:#00fff7; text-shadow:0 0 20px #00fff7;'>DNA STR Matcher</h2>
            <hr style='border:1px solid #00fff7;'>
        </div>
        """, unsafe_allow_html=True)
        st.info("""
        <b>Instructions:</b><br>
        1. Upload your DNA STR datasets.<br>
        2. Classify peaks with ML.<br>
        3. Detect unique individuals.<br>
        4. Match with criminal database.<br>
        5. Explore advanced analytics.<br>
        """, icon="ℹ️")
        st.markdown("---")
        st.markdown("<small style='color:#00fff7;'>Made with ❤️ for forensic science</small>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<b style='color:#00fff7;'>Progress</b>", unsafe_allow_html=True)
        st.progress(0)

    # Cinematic loading spinner
    with st.spinner('Initializing DNA sequencer...'):
        time.sleep(1.0)
    st.success('Sequencer ready!')

    # --- Main Workflow Tabs ---
    tabs = st.tabs([
        "Upload Data", "STR Classification", "Model Comparison", "Unique Individuals", "Matching Results", "Advanced Analytics", "AI Insights & Report"
    ])

    # --- Tab 1: Upload Data ---
    with tabs[0]:
        st.markdown("""
        <div class='glass-card'>
        <h3>Step 1: Upload DNA STR Datasets</h3>
        <div style='color:#00fff7;'>Upload your evidence and criminal STR CSV files to begin.</div>
        </div>
        """, unsafe_allow_html=True)
        evidence_file = st.file_uploader("📁 Upload DNA STR Dataset (Crime Scene)", type=["csv"], key="evidence")
        criminal_file = st.file_uploader("📂 Upload Criminal STR Dataset", type=["csv"], key="criminal")
        upload_ready = evidence_file is not None and criminal_file is not None
        if upload_ready:
            st.success("✅ Files uploaded! Proceed to the next step using the tabs above.")
        else:
            st.warning("Please upload both files to continue.")

    # --- Data Loading (shared for all tabs) ---
    def load_csv(file, name=""):
        try:
            file.seek(0)
            df = pd.read_csv(file)
        except Exception:
            st.error(f"❌ Could not load {name} file.")
            return None
        return df

    df, criminal_df = None, None
    if evidence_file and criminal_file:
        df = load_csv(evidence_file, "Evidence")
        criminal_df = load_csv(criminal_file, "Criminal")
        if df is None or criminal_df is None:
            st.stop()
        # Validate columns
        required_evidence = ['Locus', 'Allele', 'Peak_Height', 'Dye_Channel', 'Label']
        required_criminal = ['Criminal_ID', 'Locus', 'Allele']
        if not all(col in df.columns for col in required_evidence):
            st.error(f"❌ Evidence must contain: {required_evidence}")
            st.stop()
        if not all(col in criminal_df.columns for col in required_criminal):
            st.error(f"❌ Criminal file must contain: {required_criminal}")
            st.stop()

    # --- Tab 2: STR Classification ---
    with tabs[1]:
        st.markdown("<div class='glass-card'><h3>Step 2: STR Peak Classification</h3></div>", unsafe_allow_html=True)
        if df is not None:
            try:
                label_encoder = LabelEncoder()
                df['Label_encoded'] = label_encoder.fit_transform(df['Label'])
                X = df[['Peak_Height', 'Dye_Channel']]
                X = pd.get_dummies(X, columns=['Dye_Channel'])
                y = df['Label_encoded']
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                if len(set(y)) < 2:
                    st.error("❌ 'Label' column must contain at least two unique classes.")
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                    models = {
                        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
                        "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
                    }
                    results = []
                    trained_models = {}
                    for name, model in models.items():
                        with st.spinner(f"Training {name}..."):
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            acc = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            results.append({"Model": name, "Accuracy": acc, "F1-Score": f1})
                            trained_models[name] = model
                            with st.expander(f"🔍 {name} Classification Report", expanded=False):
                                st.text(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
                    st.session_state['results'] = results
                    st.session_state['trained_models'] = trained_models
                    st.success("Classification complete! Results saved. See model comparison tab for results.")
            except Exception as e:
                st.error(f"Error during classification: {e}")
        else:
            st.info("Upload data in the first tab to enable classification.")

    # --- Tab 3: Model Comparison (FIXED & ENHANCED) ---
    with tabs[2]:
        st.markdown("<div class='glass-card'><h3>Step 3: ML Model F1-Score Comparison</h3></div>", unsafe_allow_html=True)
        if st.session_state['results'] is not None:
            results_df = pd.DataFrame(st.session_state['results'])

            # Prepare percentage text labels
            results_df['F1_pct_text'] = results_df['F1-Score'].apply(lambda x: f"{x*100:.2f}%")

            # Plotly bar chart with forced 0-1 Y-axis and percentage text
            fig = px.bar(
                results_df,
                x="Model",
                y="F1-Score",
                color="Model",
                text=results_df["F1_pct_text"],
                color_discrete_sequence=px.colors.sequential.Teal,
                title="🔍 ML Models F1-Score Performance"
            )
            # display the text as-is (we passed percent strings)
            fig.update_traces(textposition='outside')
            # ✅ Force Y-axis to 0..1
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(plot_bgcolor="#0f2027", paper_bgcolor="#0f2027", font_color="#00fff7", yaxis_title="F1-Score")
            st.plotly_chart(fig, use_container_width=True)

            # Best Model Highlight
            best_idx = results_df['F1-Score'].idxmax()
            best_model = results_df.loc[best_idx, 'Model']
            best_score = results_df.loc[best_idx, 'F1-Score']
            st.success(f"🔥 Best Model Based on F1-Score: **{best_model} ({best_score*100:.2f}%)**")

            # ASCII / Text-Based proportional visualization
            st.markdown("### 📊 Text-Based Visualization")
            for _, row in results_df.iterrows():
                # scale to 20 blocks for granularity
                bar_length = int(row['F1-Score'] * 20)
                bar = "█" * bar_length
                st.markdown(f"`{row['Model']:<18} {bar:<20} {row['F1-Score']*100:6.2f}%`")
        else:
            st.info("Run classification in the previous tab to see model comparison.")

    # --- Tab 4: Unique Individuals ---
    with tabs[3]:
        st.markdown("<div class='glass-card'><h3>Step 4: Detect Unique Individuals</h3></div>", unsafe_allow_html=True)
        if df is not None:
            allele_df = df[df['Label'] == 'Allele'].sort_values(by=["Locus", "Allele"]).reset_index(drop=True)
            allele_df["Individual_ID"] = [f"IND{str(i//32 + 1).zfill(4)}" for i in range(len(allele_df))]
            individuals = defaultdict(set)
            for _, row in allele_df.iterrows():
                individuals[row["Individual_ID"]].add((row["Locus"], row["Allele"]))
            st.session_state['individuals'] = individuals
            st.session_state['allele_df'] = allele_df
            st.info(f"🧬 Unique Individuals Detected: {len(individuals)}")
            st.dataframe(allele_df[["Individual_ID", "Locus", "Allele"]])
        else:
            st.info("Upload and classify data to detect individuals.")

    # --- Tab 5: Matching Results ---
    with tabs[4]:
        st.markdown("<div class='glass-card'><h3>Step 5: Match with Criminal STRs</h3></div>", unsafe_allow_html=True)
        if df is not None and criminal_df is not None and 'allele_df' in st.session_state:
            allele_df = st.session_state['allele_df']
            allele_df = allele_df.sort_values(by=["Locus", "Allele"]).reset_index(drop=True)
            allele_df["Individual_ID"] = [f"IND{str(i//32 + 1).zfill(4)}" for i in range(len(allele_df))]
            individuals = defaultdict(set)
            for _, row in allele_df.iterrows():
                individuals[row["Individual_ID"]].add((row["Locus"], row["Allele"]))

            criminals = defaultdict(set)
            for _, row in criminal_df.iterrows():
                criminals[str(row["Criminal_ID"]).strip()].add((row["Locus"], row["Allele"]))

            matches = []
            partial_matches = []
            for ind_id, ind_alleles in individuals.items():
                for crim_id, crim_alleles in criminals.items():
                    common = len(ind_alleles & crim_alleles)
                    if common >= 4:
                        matches.append({"Individual_ID": ind_id, "Criminal_ID": crim_id, "Match_Loci": common})
                    elif common >= 2:
                        partial_matches.append({"Individual_ID": ind_id, "Potential_Criminal_ID": crim_id, "Match_Loci": common})

            # Ensure consistent columns even if empty
            match_df = pd.DataFrame(matches, columns=["Individual_ID", "Criminal_ID", "Match_Loci"])
            partial_df = pd.DataFrame(partial_matches, columns=["Individual_ID", "Potential_Criminal_ID", "Match_Loci"])

            # store into session_state so other tabs can use
            st.session_state['match_df'] = match_df
            st.session_state['partial_df'] = partial_df

            matched_ids = set(match_df["Individual_ID"].unique()) | set(partial_df["Individual_ID"].unique())
            non_matched_list = list(set(individuals.keys()) - matched_ids)
            non_matched = pd.DataFrame({"Individual_ID": non_matched_list})

            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Full Matches", len(match_df["Individual_ID"].unique()) if not match_df.empty else 0)
            col2.metric("🟡 Partial Matches", len(partial_df["Individual_ID"].unique()) if not partial_df.empty else 0)
            col3.metric("❌ No Match", len(non_matched))

            if not match_df.empty:
                st.markdown("### 🔗 Full Matches")
                st.dataframe(match_df)
                st.download_button("⬇️ Download Full Matches", match_df.to_csv(index=False), "full_matches.csv")
                unique_full_match_ids = match_df["Individual_ID"].unique()
                st.info(f"There are {len(unique_full_match_ids)} unique individuals with at least one full (exact) match to a criminal.")
                st.write("**Unique individuals with full matches:**", list(unique_full_match_ids))

            if not partial_df.empty:
                st.markdown("### 💾 Partial Matches")
                st.dataframe(partial_df)
                st.download_button("⬇️ Download Partial Matches", partial_df.to_csv(index=False), "partial_matches.csv")

            if not non_matched.empty:
                st.markdown("### ❌ No Match Individuals")
                st.dataframe(non_matched)
                st.download_button("⬇️ Download No Matches", non_matched.to_csv(index=False), "non_matches.csv")

            # Matches per Criminal chart
            if not match_df.empty:
                st.markdown("### 📊 Matches per Criminal")
                matches_per_criminal = match_df['Criminal_ID'].value_counts().reset_index()
                matches_per_criminal.columns = ['Criminal_ID', 'Number_of_Matches']
                fig2 = px.bar(
                    matches_per_criminal,
                    x='Criminal_ID',
                    y='Number_of_Matches',
                    color='Number_of_Matches',
                    color_continuous_scale='Bluered',
                    title='Number of Individuals Matched per Criminal',
                    labels={'Number_of_Matches': 'Individuals Matched'}
                )
                fig2.update_layout(plot_bgcolor='#0f2027', paper_bgcolor='#0f2027', font_color='#00fff7')
                st.plotly_chart(fig2, use_container_width=True)

            # Network Graph Visualization
            if not match_df.empty:
                st.markdown("### 🕸️ Match Network Graph")
                G = nx.Graph()
                for _, row in match_df.iterrows():
                    G.add_node(row['Individual_ID'], type='individual')
                    G.add_node(row['Criminal_ID'], type='criminal')
                    G.add_edge(row['Individual_ID'], row['Criminal_ID'], weight=row['Match_Loci'])
                pos = nx.spring_layout(G, seed=42)
                edge_x, edge_y = [], []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#00fff7'), hoverinfo='none', mode='lines')
                node_x, node_y, node_color, node_text = [], [], [], []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_color.append('#00ff99' if G.nodes[node]['type']=='criminal' else '#00fff7')
                    node_text.append(node)
                node_trace = go.Scatter(
                    x=node_x, y=node_y, mode='markers+text',
                    marker=dict(size=18, color=node_color, line=dict(width=2, color='#fff')),
                    text=node_text, textposition='top center',
                    hoverinfo='text')
                fig3 = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
                    showlegend=False, hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    plot_bgcolor='#0f2027', paper_bgcolor='#0f2027', font_color='#00fff7',
                    title='Individuals ↔ Criminals Match Network'
                ))
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Upload and process data in previous tabs to see matches.")

    # --- Tab 6: Advanced Analytics ---
    with tabs[5]:
        st.markdown("<div class='glass-card'><h3>Step 6: Advanced Analytics</h3></div>", unsafe_allow_html=True)
        if not st.session_state['match_df'].empty:
            match_df = st.session_state['match_df']
            st.markdown("### 🧬 Distribution of Loci Matches")
            fig4 = px.histogram(
                match_df,
                x='Match_Loci',
                nbins=10,
                color='Match_Loci',
                title='Distribution of Loci Matches in Full Matches',
                color_discrete_sequence=px.colors.sequential.Teal
            )
            fig4.update_layout(plot_bgcolor='#0f2027', paper_bgcolor='#0f2027', font_color='#00fff7')
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No full matches available for loci distribution.")

        if df is not None and criminal_df is not None:
            st.markdown("### 🧬 Allele Frequency Comparison")
            evidence_allele_counts = df['Allele'].value_counts().reset_index()
            evidence_allele_counts.columns = ['Allele', 'Evidence_Count']
            criminal_allele_counts = criminal_df['Allele'].value_counts().reset_index()
            criminal_allele_counts.columns = ['Allele', 'Criminal_Count']
            merged_counts = pd.merge(evidence_allele_counts, criminal_allele_counts, on='Allele', how='outer').fillna(0)
            fig5 = go.Figure()
            fig5.add_trace(go.Bar(x=merged_counts['Allele'], y=merged_counts['Evidence_Count'], name='Evidence', marker_color='#00fff7'))
            fig5.add_trace(go.Bar(x=merged_counts['Allele'], y=merged_counts['Criminal_Count'], name='Criminal', marker_color='#00ff99'))
            fig5.update_layout(barmode='group', title='Allele Frequency: Evidence vs. Criminal',
                            plot_bgcolor='#0f2027', paper_bgcolor='#0f2027', font_color='#00fff7')
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Upload and process data to see advanced analytics.")

    # --- Tab 7: AI Insights & Report ---
    with tabs[6]:
        st.markdown("<div class='glass-card'><h3>Step 7: AI Insights & Downloadable Report</h3></div>", unsafe_allow_html=True)
        if df is not None and criminal_df is not None:
            st.markdown("#### 🧠 AI Insights")
            match_df = st.session_state['match_df']
            partial_df = st.session_state['partial_df']
            if not match_df.empty:
                top_criminal = match_df['Criminal_ID'].value_counts().idxmax()
                st.success(f"Most matched criminal: {top_criminal}")
                st.info(f"Total unique individuals detected (rough): {len(st.session_state['individuals'])}")
                st.info(f"Total full matches: {len(match_df['Individual_ID'].unique())}")
                st.info(f"Total partial matches: {len(partial_df['Individual_ID'].unique()) if not partial_df.empty else 0}")
            else:
                st.warning("No full matches found.")

            # Downloadable HTML report
            st.markdown("#### 📄 Downloadable HTML Report")
            html_report = f"""
            <html><head><title>DNA STR Matcher Movie Mode Report</title></head><body style='background:#0f2027;color:#00fff7;font-family:Orbitron,Share Tech Mono,monospace;'>
            <h1>DNA STR Matcher Movie Mode Report</h1>
            <h2>Summary</h2>
            <ul>
            <li>Most matched criminal: {top_criminal if 'top_criminal' in locals() else 'N/A'}</li>
            <li>Total unique individuals (rough): {len(st.session_state['individuals'])}</li>
            <li>Total full matches: {len(match_df['Individual_ID'].unique()) if not match_df.empty else 0}</li>
            <li>Total partial matches: {len(partial_df['Individual_ID'].unique()) if not partial_df.empty else 0}</li>
            </ul>
            </body></html>
            """
            b64 = base64.b64encode(html_report.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64}" download="dna_str_report.html">⬇️ Download HTML Report</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("Upload and process data to see AI insights and download reports.")

    # --- Accessibility & Theme Toggle ---
    st.markdown("""
    <script>
    function setTheme(theme) {
        document.body.setAttribute('data-theme', theme);
    }
    </script>
    <div style='position:fixed;bottom:24px;right:24px;z-index:999;'>
        <button onclick="setTheme('light')" style="margin-right:8px;padding:8px 16px;border-radius:8px;border:none;background:#fff;color:#00fff7;font-weight:bold;">☀️ Light</button>
        <button onclick="setTheme('dark')" style="padding:8px 16px;border-radius:8px;border:none;background:#1a1a40;color:#fff;font-weight:bold;">🌙 Dark</button>
    </div>
    """, unsafe_allow_html=True)
