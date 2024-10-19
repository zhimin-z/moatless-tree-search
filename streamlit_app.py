import argparse
import json
import logging
import os
import sys
import pandas as pd

import streamlit as st
from dotenv import load_dotenv

from moatless.benchmark.report import generate_report
from moatless.streamlit.shared import trajectory_table
from moatless.search_tree import SearchTree
from moatless.streamlit.tree_vizualization import update_visualization


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(layout="wide", page_title="Moatless Visualizer", initial_sidebar_state="collapsed")

container = st.container()

def reset_cache():
    st.cache_data.clear()
    st.session_state.root_node = None
    st.session_state.selected_node_id = None
    st.info("Cache cleared and session state reset")

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

def get_search_tree(tree_path: str) -> SearchTree:
    tree = SearchTree.from_file(tree_path, workspace=None)
    logger.info(f"Loaded search tree from {tree_path} with {len(tree.root.get_all_nodes())} nodes")
    return tree

def tree_table(directory_path: str):
    tree_files = [f for f in os.listdir(directory_path) if f.endswith('.json') and f != 'report.json']
    
    if not tree_files:
        st.warning("No tree files found in the selected directory.")
        return

    data = []
    for tree_file in tree_files:
        tree_path = os.path.join(directory_path, tree_file)
        with open(tree_path, 'r') as f:
            tree_data = json.load(f)
        
        metadata = tree_data.get('metadata', {})
        instance_id = metadata.get('instance_id', 'Unknown')
        
        data.append({
            "Instance ID": instance_id,
            "File": tree_file,
            "Path": tree_path
        })

    df = pd.DataFrame(data)
    st.dataframe(df)

    selected_indices = st.multiselect("Select trees to visualize:", df.index)
    if selected_indices:
        selected_tree_path = df.loc[selected_indices[0], "Path"]
        st.session_state.search_tree = get_search_tree(selected_tree_path).root
        st.rerun()

if __name__ == "__main__":
    args = parse_args()
    moatless_dir = args.moatless_dir

    st.sidebar.text(f"Using Moatless directory: {moatless_dir}")

    directories = [d for d in os.listdir(moatless_dir) if os.path.isdir(os.path.join(moatless_dir, d))]
    directories.sort()

    selected_directory = st.selectbox("Select a directory", directories)

    if selected_directory:
        selected_directory_path = os.path.join(moatless_dir, selected_directory)
        if st.button("Regenerate Report"):
            with st.spinner("Generating report..."):
                generate_report(selected_directory_path)
            st.success("Report generated successfully!")
            st.rerun()

        # Check for tree_path in query params
        if "trajectory_path" in st.query_params:
            selected_tree_path = st.query_params["trajectory_path"]
            if "search_tree" not in st.session_state: # TODO or st.session_state.root_node._persist_path != selected_tree_path:
                with st.spinner(f"Loading search tree: {selected_tree_path}"):
                    st.session_state.search_tree = get_search_tree(selected_tree_path)
                    st.session_state.selected_tree_path = selected_tree_path

        if "search_tree" in st.session_state:
            update_visualization(container, st.session_state.search_tree, selected_tree_path)
        else:
            trajectory_table(selected_directory_path)

    if st.button("Reset Cache"):
        reset_cache()
