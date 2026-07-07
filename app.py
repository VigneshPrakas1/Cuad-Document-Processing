import streamlit as st
import json
import tempfile
from src.pipeline import process_uploaded_contract

st.set_page_config(
    page_title="AI Contract Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Contract Analyzer")
st.caption("AI-powered Legal Contract Understanding")

# ---------------- Sidebar ----------------

with st.sidebar:

    import os
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=120)

    st.header("Pipeline")

    st.code("""
PDF
 ↓
Chunking
 ↓
LLM
 ↓
Clause Extraction
 ↓
Summary
""")

    st.divider()

    st.write("Version 1.0")

# ---------------- Upload ----------------

uploaded_file = st.file_uploader(
    "Upload Contract",
    type=["pdf","txt"]
)

if uploaded_file:

    st.success(f"{uploaded_file.name} uploaded")

    if st.button(
        "🚀 Analyze Contract",
        use_container_width=True
    ):

        progress = st.progress(0)
        
        progress.progress(20)
        # load pdf
        
        progress.progress(40)
        # chunk
        
        progress.progress(60)
        # LLM
        
        progress.progress(80)
        # summary
        
        # your pipeline
        suffix = uploaded_file.name.split(".")[-1]
        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix="."+suffix
        ) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name
            
        try:
            result = process_uploaded_contract(temp_path)
        except Exception as e:
            if "RateLimit" in str(e) or "429" in str(e):
                st.error(
                    "🚫 Groq API quota exceeded.\n\n"
                    "Please wait for your quota to reset, "
                    "switch to mock mode, or use another LLM provider."
                )
                st.stop()
            else:
                st.exception(e)
        
        progress.progress(100)
        progress.empty()

        st.success("Analysis Complete")

        st.divider()

        # ---------- Metrics ----------

        status_cols = st.columns(4)

        status_cols[0].success("Summary Generated")
        status_cols[1].metric("Termination", "Found" if result["termination_clause"]["found"] else "Missing")
        status_cols[2].metric("Confidentiality", "Found" if result["confidentiality_clause"]["found"] else "Missing")
        status_cols[3].metric("Liability", "Found" if result["liability_clause"]["found"] else "Missing")

        st.divider()

        # ---------- Tabs ----------

        summary_tab,term_tab,conf_tab,lia_tab = st.tabs([
            "📄 Summary",
            "🟥 Termination",
            "🔒 Confidentiality",
            "⚖ Liability"
        ])

        with summary_tab:

            st.markdown("### Contract Summary")

            st.write(result["summary"])

        with term_tab:

            with st.expander(
                "Termination Clause",
                expanded=True
            ):
                clause = result["termination_clause"]
                st.write(clause["notes"])
                if clause["excerpts"]:
                    st.markdown("### Excerpts")
                    for e in clause["excerpts"]:
                        st.code(e)

        with conf_tab:

            with st.expander(
                "Confidentiality Clause",
                expanded=True
            ):
                clause = result["confidentiality_clause"]
                st.write(clause["notes"])
                if clause["excerpts"]:
                    st.markdown("### Excerpts")
                    for e in clause["excerpts"]:
                        st.code(e)

        with lia_tab:

            with st.expander(
                "Liability Clause",
                expanded=True
            ):
                clause = result["liability_clause"]
                st.write(clause["notes"])
                if clause["excerpts"]:
                    st.markdown("### Excerpts")
                    for e in clause["excerpts"]:
                        st.code(e)

        st.divider()

        st.download_button(
            "📥 Download JSON",
            json.dumps(result, indent=4),
            file_name="analysis.json"
        )