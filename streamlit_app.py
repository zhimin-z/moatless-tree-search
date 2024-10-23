import argparse
import json
import logging
import os
import sys
import pandas as pd

import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.uploaded_file_manager import UploadedFile

from moatless.benchmark.report import generate_report
from moatless.streamlit.shared import trajectory_table
from moatless.search_tree import SearchTree
from moatless.streamlit.tree_vizualization import update_visualization


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(layout="wide", page_title="Moatless Visualizer", initial_sidebar_state="collapsed")

container = st.container()

def parse_args():
    parser = argparse.ArgumentParser(description="Moatless Visualizer")
    parser.add_argument("--moatless_dir", default=os.getenv("MOATLESS_DIR", "/tmp/moatless"),
                        help="Directory for Moatless data (default: $MOATLESS_DIR or /tmp/moatless)")

    if '--' in sys.argv:
        # Remove streamlit-specific arguments
        streamlit_args = sys.argv[sys.argv.index('--') + 1:]
        args = parser.parse_args(sys.argv[1:sys.argv.index('--')])
    else:
        args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    
    if "path" in st.query_params:
        file_path = st.query_params["path"]
    else:
        file_path = None

    if not file_path:
        st.sidebar.text("Please provide a file path.")
        file_path = st.text_input("Enter the full path to your JSON file:")
        load_button = st.button("Load File")

        if load_button and file_path:
            # Set the path as a query parameter and rerun
            st.query_params["path"] = file_path
            st.rerun()
    else:
        st.sidebar.text(f"Loading file: {file_path}")

    if file_path:
        if os.path.exists(file_path):
            file_name = os.path.basename(file_path).lower()
            
            if file_name == "report.json":
                # Handle as a table
                with st.spinner("Loading report..."):
                    df = pd.read_json(file_path)
                    trajectory_table(df)
            else:
                # Handle as trajectory_path
                with st.spinner("Loading search tree from trajectory file"):
                    st.session_state.search_tree = SearchTree.from_file(file_path)
                    st.session_state.selected_tree_path = file_name
                    update_visualization(container, st.session_state.search_tree, st.session_state.selected_tree_path)
        else:
            st.error("The specified file does not exist. Please check the path and try again.")

    if not file_path:
        st.info("Please provide a valid file path and click 'Load File' to begin.")
