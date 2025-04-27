import streamlit as st
import os
import json
import pandas as pd
from utils import (
    extract_fragments, perform_clustering, match_patterns,
    structure_clusters, generate_analysis_report
)
import tempfile
import zipfile
import time
from bs4 import BeautifulSoup
import shutil
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Set page config
st.set_page_config(
    page_title="HTML Fragment Analysis",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'pattern_data' not in st.session_state:
    st.session_state.pattern_data = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None
if 'noise' not in st.session_state:
    st.session_state.noise = None
if 'report_data' not in st.session_state:
    st.session_state.report_data = None
if 'processing_progress' not in st.session_state:
    st.session_state.processing_progress = 0

def load_latest_report():
    """Load the most recent report file."""
    report_files = glob.glob('clustered_output/report_*.json')
    if not report_files:
        return None
    
    # Get the most recent file based on modification time
    latest_report = max(report_files, key=os.path.getmtime)
    
    try:
        with open(latest_report, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading report file: {str(e)}")
        return None

def process_zip_chunk(args):
    """Process a chunk of files from the ZIP archive."""
    zip_path, start_idx, end_idx = args
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()[start_idx:end_idx]
            combined_html = ""
            fragment_data = []
            
            for file in files:
                if file.endswith('.html') or file.endswith('.xhtml'):
                    with zip_ref.open(file) as f:
                        html_content = f.read().decode('utf-8')
                        soup = BeautifulSoup(html_content, 'html.parser')
                        for section in soup.find_all('section'):
                            fragment_info = {
                                'html': section.prettify(),
                                'original_file': file,
                                'line_number': section.sourceline
                            }
                            fragment_data.append(fragment_info)
                            combined_html += section.prettify()
            return combined_html, fragment_data
    except Exception as e:
        st.error(f"Error processing zip chunk: {str(e)}")
        return "", []

def process_large_zip_file(zip_path, pattern_data, progress_bar):
    """Process a large ZIP file in chunks to handle memory efficiently."""
    temp_dir = tempfile.mkdtemp()
    try:
        # Process ZIP file in chunks using multiprocessing
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            chunk_size = max(1, len(files) // (multiprocessing.cpu_count() * 2))
            chunks = [(zip_path, i, min(i + chunk_size, len(files))) 
                     for i in range(0, len(files), chunk_size)]
        
        combined_html = []
        fragment_sources = []
        total_chunks = len(chunks)
        
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
            for i, (html_chunk, fragment_chunk) in enumerate(executor.map(process_zip_chunk, chunks)):
                combined_html.append(html_chunk)
                fragment_sources.extend(fragment_chunk)
                
                # Update progress
                progress = (i + 1) / total_chunks * 100
                st.session_state.processing_progress = progress
                progress_bar.progress(int(progress))
        
        if not combined_html:
            st.error("No valid HTML files processed")
            return None, None, None
        
        # Process fragments
        initial_fragment_id = 0
        fragment_name = os.path.splitext(os.path.basename(zip_path))[0]
        soup = BeautifulSoup(''.join(combined_html), 'html.parser')
        initial_parent_tag = soup.find('section').name if soup.find('section') else 'body'
        
        fragment_data, _ = extract_fragments(
            ''.join(combined_html), initial_fragment_id, fragment_name,
            initial_parent_tag, fragment_sources
        )
        
        if not fragment_data:
            st.error("No valid HTML files processed")
            return None, None, None
        
        # Create DataFrame and perform clustering
        columns = ["fragment_id", "fragment_name", "Level", "element_type", "element_class",
                  "attributes", "structure", "original_file", "original_line"]
        df_combined = pd.DataFrame(fragment_data, columns=columns)
        
        df_combined = match_patterns(df_combined, pattern_data)
        df_clustered = perform_clustering(df_combined)
        
        clusters, noise = structure_clusters(df_clustered)
        
        return df_clustered, clusters, noise
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

def process_uploaded_files(zip_file, pattern_data):
    """Handle file upload and processing with progress tracking."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        # Write uploaded file to temporary file
        tmp_file.write(zip_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        st.session_state.processing_progress = 0
        
        # Process the file
        result = process_large_zip_file(tmp_file_path, pattern_data, progress_bar)
        
        # Update progress to 100%
        st.session_state.processing_progress = 100
        progress_bar.progress(100)
        
        return result
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)

def display_noise_analysis():
    st.title("Noise Fragment Analysis")
    
    # Try to load the latest report if not in session state
    if st.session_state.report_data is None:
        st.session_state.report_data = load_latest_report()
    
    if st.session_state.report_data is None:
        st.warning("No analysis data available. Please upload and process files first.")
        return
    
    report = st.session_state.report_data
    
    # Overall statistics
    st.header("Overall Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Fragments", report["overall_statistics"]["total_fragments"])
    with col2:
        st.metric(
            "Matched Fragments",
            f"{report['overall_statistics']['matched_fragments']['count']} "
            f"({report['overall_statistics']['matched_fragments']['percentage']}%)"
        )
    with col3:
        st.metric(
            "Unmatched Fragments",
            f"{report['overall_statistics']['unmatched_fragments']['count']} "
            f"({report['overall_statistics']['unmatched_fragments']['percentage']}%)"
        )
    
    # Filter controls
    st.header("Filter Fragments")
    col1, col2 = st.columns(2)
    with col1:
        show_matching = st.checkbox("Show fragments with matching patterns", value=True)
    with col2:
        show_non_matching = st.checkbox("Show fragments with no matching patterns", value=False)
    
    # Filter fragments with improved logic
    filtered_fragments = []
    for f in report["noise_fragment_analysis"]["fragments"]:
        has_pattern_comparison = f.get("pattern_comparison") is not None
        has_matching_patterns = False
        
        if has_pattern_comparison:
            if isinstance(f["pattern_comparison"], list):
                has_matching_patterns = len(f["pattern_comparison"]) > 0
            else:
                has_matching_patterns = False
        
        if (show_matching and has_matching_patterns) or (show_non_matching and not has_matching_patterns):
            filtered_fragments.append(f)
    
    if not filtered_fragments:
        st.info("No fragments match the current filter criteria.")
        return
    
    # Initialize current fragment index in session state if not exists
    if 'current_fragment_index' not in st.session_state:
        st.session_state.current_fragment_index = 0
    
    # Navigation controls
    st.header("Fragment Details")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("Previous Fragment") and st.session_state.current_fragment_index > 0:
            st.session_state.current_fragment_index -= 1
    with col2:
        st.write(f"Fragment {st.session_state.current_fragment_index + 1} of {len(filtered_fragments)}")
    with col3:
        if st.button("Next Fragment") and st.session_state.current_fragment_index < len(filtered_fragments) - 1:
            st.session_state.current_fragment_index += 1
    
    # Display current fragment
    fragment = filtered_fragments[st.session_state.current_fragment_index]
    
    # Basic info
    st.subheader("Fragment Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**ID:**", fragment["fragment_id"])
        st.write("**Element Type:**", fragment["element_type"])
    with col2:
        st.write("**Source File:**", fragment["source_file"])
        st.write("**Starting Line:**", fragment["starting_line_number"])
    
    # Tag structure
    st.subheader("Tag Structure")
    tag_cols = st.columns(4)
    for j, tag in enumerate(fragment["tag_structure"]):
        with tag_cols[j % 4]:
            st.info(f"Level {tag['level']}: {tag['tag']}")
    
    # HTML structure
    st.subheader("HTML Structure")
    st.code("\n".join(fragment["html_structure"]), language="html")
    
    # Pattern comparison
    st.subheader("Pattern Comparison")
    if fragment.get("pattern_comparison"):
        if isinstance(fragment["pattern_comparison"], list) and len(fragment["pattern_comparison"]) > 0:
            # Use tabs for pattern comparisons
            pattern_tabs = st.tabs([f"Pattern {i+1}" for i in range(len(fragment["pattern_comparison"]))])
            for tab, pattern in zip(pattern_tabs, fragment["pattern_comparison"]):
                with tab:
                    st.write(f"**Pattern Name:** {pattern['pattern_name']}")
                    
                    if pattern.get("pattern_tag_structure"):
                        st.write("**Pattern Tag Structure:**")
                        pattern_cols = st.columns(4)
                        for j, tag in enumerate(pattern["pattern_tag_structure"]):
                            with pattern_cols[j % 4]:
                                st.success(f"Level {tag['level']}: {tag['tag']}")
                    
                    if pattern.get("mismatch_details"):
                        if pattern["mismatch_details"].get("class_mismatch"):
                            st.write("**Class Mismatch:**")
                            st.write(f"Fragment: {pattern['mismatch_details']['class_mismatch']['fragment_class']}")
                            st.write(f"Pattern: {pattern['mismatch_details']['class_mismatch']['pattern_class']}")
                        
                        if pattern["mismatch_details"].get("tag_structure_mismatches"):
                            st.write("**Tag Structure Mismatches:**")
                            for mismatch in pattern["mismatch_details"]["tag_structure_mismatches"]:
                                st.write(f"Level {mismatch['level']}: {mismatch['tag']} - "
                                        f"Fragment: {mismatch['fragment_value']}, "
                                        f"Pattern: {mismatch['pattern_value']}")
        else:
            st.warning("No matching patterns found for this fragment type.")
    else:
        st.warning("No pattern comparison available for this fragment.")

def main():
    st.title("HTML Fragment Analysis")
    
    # Step 1: Upload Pattern Files
    st.header("Step 1: Upload Pattern Files")
    pattern_files = st.file_uploader(
        "Select HTML Pattern Files",
        type=['html'],
        accept_multiple_files=True
    )
    
    if pattern_files:
        pattern_data = []
        for file in pattern_files:
            html_content = file.read().decode('utf-8')
            fragment_data, _ = extract_fragments(html_content, 0, file.name, 'body')
            pattern_data.extend(fragment_data)
        
        if pattern_data:
            st.session_state.pattern_data = pd.DataFrame(pattern_data)
            st.success(f"Successfully processed {len(pattern_files)} pattern files")
    
    # Step 2: Upload and Process ZIP File
    st.header("Step 2: Upload and Process ZIP File")
    zip_file = st.file_uploader(
        "Select ZIP File (up to 2GB)", 
        type=['zip'],
        accept_multiple_files=False
    )
    
    if zip_file and st.session_state.pattern_data is not None:
        if st.button("Process Files"):
            with st.spinner("Processing files..."):
                start_time = time.time()
                df_clustered, clusters, noise = process_uploaded_files(
                    zip_file, st.session_state.pattern_data
                )
                
                if df_clustered is not None:
                    # Generate report
                    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
                        report_path = generate_analysis_report(
                            df_clustered, clusters, noise, temp_file.name,
                            st.session_state.pattern_data
                        )
                        
                        with open(report_path, 'r') as f:
                            st.session_state.report_data = json.load(f)
                    
                    processing_time = time.time() - start_time
                    st.success(f"Processing completed in {processing_time:.2f} seconds")
                    
    
    # Display noise analysis
    if st.session_state.report_data is not None:
        display_noise_analysis()
    else:
        # Try to load existing report
        st.session_state.report_data = load_latest_report()
        if st.session_state.report_data is not None:
            st.info("Loaded existing report data.")
            display_noise_analysis()

if __name__ == "__main__":
    main() 
