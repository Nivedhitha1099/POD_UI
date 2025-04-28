import streamlit as st
import os
import uuid
import zipfile
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, Tag
from sklearn.cluster import DBSCAN
from concurrent.futures import ThreadPoolExecutor
import tempfile
import re
import time

levels = range(1, 14)
decorative_tags = ['em', 'strong', 'i', 'br', 'span', 'dfn']



def extract_attributes(tag):
    if not isinstance(tag, Tag):
        raise TypeError("Expected a BeautifulSoup Tag object")
    if tag:
        attributes = []
        for attr, val in tag.attrs.items():
            if isinstance(val, list):
                val = " ".join(val)
            attributes.append(f'{attr}="{val}"')
        return ", ".join(attributes)
    return ""

def extract_unique_body_tags(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    body_tag = soup.body
    unique_tags = set()
    if body_tag:
        for tag in body_tag.find_all(recursive=True):
            unique_tags.add(tag.name)
    return list(unique_tags)

def extract_fragments(html_content, initial_fragment_id, fragment_name, initial_parent_tag, fragment_sources=None):
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        data = []
        fragment_id = initial_fragment_id
        sections = soup.find_all('section')

        def process_section(section, idx):
            local_data = []
            local_fragment_id = fragment_id + idx + 1
            parent_tag = initial_parent_tag
            original_file = None
            original_line = None
            if fragment_sources and idx < len(fragment_sources):
                original_file = fragment_sources[idx].get('original_file')
                original_line = fragment_sources[idx].get('line_number')
            for child in section.children:
                if child.name and child.name not in decorative_tags:
                    parse_element(
                        child, 1, str(uuid.uuid4()), local_data, fragment_name,
                        parent_tag, None, original_file, original_line
                    )
            return local_data

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_section, sections, range(len(sections))))

        for section_data in results:
            data.extend(section_data)

        fragment_id += len(sections)
        return data, fragment_id
    except Exception as e:
        st.error(f"Error processing fragment {fragment_id}: {e}")
        return [], initial_fragment_id

def parse_element(element, level, fragment_id, data, fragment_name, parent_tag,
                 parent_classes=None, original_file=None, original_line=None):
    if parent_classes is None:
        parent_classes = set()
    presence_encoding = {f'Level_{lvl}_{tag}': 0 for lvl in levels for tag in tags}
    presence_encoding[f'Level_{level}_{element.name}'] = 1
    element_type = element.name
    element_classes = set(element.get('class', []))
    filtered_classes = element_classes - parent_classes
    attributes = extract_attributes(element)
    fragment_data = {
        'fragment_id': fragment_id,
        'fragment_name': fragment_name,
        'Level': level,
        'element_type': element_type,
        'element_class': " ".join(filtered_classes),
        'attributes': attributes,
        'structure': element.prettify(),
        'original_file': original_file,
        'original_line': original_line,
        **presence_encoding
    }
    data.append(fragment_data)
    for child in element.children:
        if child.name and child.name not in decorative_tags:
            new_level = level + 1
            parse_child_element(child, new_level, fragment_id, data, fragment_name,
                              element.name, element_classes, original_file, original_line)

def parse_child_element(element, level, fragment_id, data, fragment_name, parent_tag,
                       parent_classes, original_file=None, original_line=None):
    presence_encoding = data[-1]
    presence_encoding[f'Level_{level}_{element.name}'] = 1
    element_classes = set(element.get('class', []))
    filtered_classes = element_classes - parent_classes
    presence_encoding['attributes'] += "; " + extract_attributes(element)
    for child in element.children:
        if child.name and child.name not in decorative_tags:
            new_level = level + 1
            parse_child_element(child, new_level, fragment_id, data, fragment_name,
                              element.name, element_classes, original_file, original_line)

def perform_clustering(df):
    enc_cols = [col for col in df.columns if col.startswith("Level_")]
    if not enc_cols:
        df['cluster'] = -1
        return df
    X = df[enc_cols].values
    dbscan = DBSCAN(eps=0.8, min_samples=2, metric='cosine')
    df['cluster'] = dbscan.fit_predict(X)
    return df

pd.set_option('future.no_silent_downcasting', True)
def compare_fragment(row, pattern_data, encoding_cols_combined, encoding_cols_pattern):
    row_encoding = row[encoding_cols_combined].fillna('')
    for _, pattern_row in pattern_data.iterrows():
        pattern_encoding = pattern_row[encoding_cols_pattern].fillna('')
        if np.array_equal(row_encoding.values, pattern_encoding.values):
            element_class = pattern_row.get('element_class', None)
            if not element_class or pd.isna(element_class):
                element_class = pattern_row.get('fragment_name', 'Unknown')
            return element_class, 1
    return None, 0

def match_patterns(df, patterns):
    if isinstance(patterns, list):
        patterns = pd.DataFrame(patterns)
    if df.empty or patterns.empty:
        df['element_class'] = None
        df['match'] = 0
        return df
    presence_encoding_cols_combined = [col for col in df.columns if col.startswith("Level_")]
    presence_encoding_cols_pattern = [col for col in patterns.columns if col.startswith("Level_")]
    if not presence_encoding_cols_combined or not presence_encoding_cols_pattern:
        df['element_class'] = None
        df['match'] = 0
        return df
    try:
        df[['element_class', 'match']] = df.apply(
            lambda row: pd.Series(compare_fragment(row, patterns, presence_encoding_cols_combined, presence_encoding_cols_pattern)),
            axis=1
        )
    except Exception as e:
        st.error(f"Error in match_patterns: {str(e)}")
        df['element_class'] = None
        df['match'] = 0
    return df

def extract_zip_and_combine_chapters(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tempfile.gettempdir())
        combined_html = []
        fragment_data = []
        
        # Get all HTML files
        html_files = []
        for root, dirs, files in os.walk(tempfile.gettempdir()):
            if os.path.basename(root).lower().startswith('chapter'):
                for file in files:
                    if file.endswith('.html') or file.endswith('.xhtml'):
                        html_files.append(os.path.join(root, file))
        
        # Process files in parallel
        def process_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    file_fragments = []
                    for section in soup.find_all('section'):
                        fragment_info = {
                            'html': section.prettify(),
                            'original_file': file_path,
                            'line_number': section.sourceline
                        }
                        file_fragments.append(fragment_info)
                    return section.prettify(), file_fragments
            except Exception as e:
                st.error(f"Error processing file {file_path}: {e}")
                return "", []
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_file, html_files))
        
        for html_chunk, fragments in results:
            combined_html.append(html_chunk)
            fragment_data.extend(fragments)
            
        return "".join(combined_html), fragment_data
    except Exception as e:
        st.error(f"Error extracting and processing ZIP file: {e}")
        return None, []

def generate_analysis_report(df_clustered, clusters, noise, pattern_data=None):
    report = {
        "overall_statistics": {},
        "cluster_statistics": {},
        "element_class_distribution": {},
        "noise_fragment_analysis": {
            "total_noise_fragments": len(noise),
            "fragments": []
        }
    }
    total_fragments = len(df_clustered)
    matched_fragments = int(df_clustered['match'].sum())
    unmatched_fragments = total_fragments - matched_fragments
    report["overall_statistics"] = {
        "total_fragments": total_fragments,
        "matched_fragments": {
            "count": matched_fragments,
            "percentage": round(matched_fragments/total_fragments*100, 2)
        },
        "unmatched_fragments": {
            "count": unmatched_fragments,
            "percentage": round(unmatched_fragments/total_fragments*100, 2)
        }
    }
    cluster_counts = {}
    for cluster_name, cluster_data in clusters.items():
        cluster_counts[cluster_name] = sum(len(fragments) for fragments in cluster_data.values())
    report["cluster_statistics"] = {
        "number_of_clusters": len(clusters),
        "clusters": [
            {"name": name, "fragment_count": count}
            for name, count in cluster_counts.items()
        ]
    }
    class_distribution = df_clustered['element_class'].value_counts()
    report["element_class_distribution"] = [
        {
            "class_name": "No class assigned" if pd.isna(class_name) else class_name,
            "count": int(count)
        }
        for class_name, count in class_distribution.items()
    ]
    for i, fragment in enumerate(noise):
        fragment_id = fragment.get('fragment_id', 'Unknown')
        fragment_name = fragment.get('fragment_name', 'Unknown')
        element_type = fragment.get('element_type', 'Unknown')
        original_file = fragment.get('original_file', 'Unknown')
        original_line = fragment.get('original_line', 'Unknown')
        if original_file and original_file != 'Unknown':
            original_file = os.path.basename(original_file)
        
        # Get encoding fields
        encoding_fields = []
        for key, value in fragment.items():
            if isinstance(key, str) and key.startswith('Level_') and ((isinstance(value, dict) and value.get('value') == 1) or value == 1):
                level, tag = key.split('_')[1:]
                encoding_fields.append({
                    "level": int(level),
                    "tag": tag
                })
        
        # Get HTML structure
        starting_line = fragment.get('original_line')
        if starting_line and starting_line != 'Unknown' and isinstance(starting_line, (int, float, str)):
            try:
                starting_line = int(float(starting_line))
                if pd.isna(starting_line):
                    starting_line = 1
            except (ValueError, TypeError):
                starting_line = 1
        else:
            starting_line = 1
        
        structure = fragment.get('structure', '')
        lines = structure.split('\\n')
        html_structure = [
            f"Line {line_num}: {line.strip()}"
            for line_num, line in enumerate(lines, starting_line)
            if line.strip()
        ]
        
        # Pattern comparison analysis
        pattern_comparison = []
        if pattern_data is not None:
            filtered_patterns = pattern_data[pattern_data['element_type'] == element_type]
            for _, pattern in filtered_patterns.iterrows():
                pattern_info = {
                    "pattern_name": pattern.get('fragment_name', 'Unknown'),
                    "pattern_tag_structure": []
                }
                
                # Get pattern tag structure
                for key, value in pattern.items():
                    if isinstance(key, str) and key.startswith('Level_') and value == 1:
                        level, tag = key.split('_')[1:]
                        pattern_info["pattern_tag_structure"].append({
                            "level": int(level),
                            "tag": tag
                        })
                
                # Check for mismatches
                mismatch_details = {}
                
                # Check class mismatch
                fragment_class = fragment.get('element_class', '')
                pattern_class = pattern.get('element_class', '')
                if fragment_class != pattern_class:
                    mismatch_details["class_mismatch"] = {
                        "fragment_class": fragment_class,
                        "pattern_class": pattern_class
                    }
                
                # Check tag structure mismatches
                tag_structure_mismatches = []
                for pattern_tag in pattern_info["pattern_tag_structure"]:
                    found = False
                    for fragment_tag in encoding_fields:
                        if pattern_tag["level"] == fragment_tag["level"] and pattern_tag["tag"] == fragment_tag["tag"]:
                            found = True
                            break
                    if not found:
                        tag_structure_mismatches.append({
                            "level": pattern_tag["level"],
                            "tag": pattern_tag["tag"]
                        })
                
                if tag_structure_mismatches:
                    mismatch_details["tag_structure_mismatches"] = tag_structure_mismatches
                
                if mismatch_details:
                    pattern_info["mismatch_details"] = mismatch_details
                
                pattern_comparison.append(pattern_info)
        
        noise_fragment = {
            "fragment_number": i+1,
            "fragment_id": fragment_id,
            "element_type": element_type,
            "source_file": original_file,
            "starting_line_number": starting_line,
            "tag_structure": encoding_fields,
            "html_structure": html_structure,
            "pattern_comparison": pattern_comparison
        }
        report["noise_fragment_analysis"]["fragments"].append(noise_fragment)
    return report

def structure_clusters(df):
    clusters = {}
    noise = []
    element_types = set(df['element_type'].unique())
    for element_type in element_types:
        cluster_name = f"cluster_{element_type}"
        cluster_groups = {}
        cluster_data = df[df['element_type'] == element_type]
        for _, fragment in cluster_data.iterrows():
            element_class = fragment['element_class']
            level_encodings = {}
            for attr in df.columns:
                if attr.startswith("Level_") and fragment[attr] == 1:
                    classes = extract_classes_from_structure(fragment['structure'], attr.split('_')[2], int(attr.split('_')[1]))
                    level_encodings[attr] = {
                        "value": 1,
                        "class": classes
                    }
            if pd.isna(element_class):
                noise_fragment = {
                    "fragment_id": fragment['fragment_id'],
                    "fragment_name": fragment['fragment_name'],
                    "Level": fragment['Level'],
                    "element_type": fragment['element_type'],
                    "element_class": "",
                    "full_attributes": fragment['attributes'],
                    "structure": fragment['structure'],
                    "original_file": fragment.get('original_file', None),
                    "original_line": fragment.get('original_line', None),
                    **level_encodings
                }
                noise.append(noise_fragment)
                continue
            filtered_fragment = {
                "fragment_id": fragment['fragment_id'],
                "fragment_name": fragment['fragment_name'],
                "Level": fragment['Level'],
                "element_type": fragment['element_type'],
                "element_class": element_class or "",
                "full_attributes": fragment['attributes'],
                "structure": fragment['structure'],
                "original_file": fragment.get('original_file', None),
                "original_line": fragment.get('original_line', None),
                **level_encodings
            }
            group_key = element_class if element_class else "unknown"
            if group_key not in cluster_groups:
                cluster_groups[group_key] = []
            cluster_groups[group_key].append(filtered_fragment)
        if cluster_groups:
            clusters[cluster_name] = cluster_groups
    return clusters, noise

def extract_classes_from_structure(structure, element_type, level):
    soup = BeautifulSoup(structure, 'html.parser')
    classes = []
    def process_element(element, current_level):
        if current_level == level:
            if element.name == element_type:
                classes.extend([cls.strip() for cls in element.get('class', [])])
        elif current_level < level:
            for child in element.children:
                if isinstance(child, Tag):
                    process_element(child, current_level + 1)
    for child in soup.children:
        if isinstance(child, Tag):
            process_element(child, 1)
    return sorted(set(classes))

# Global variable to hold tags extracted from pattern files
tags = []

def main():
    st.title("HTML Fragment Analysis - Streamlit App")

    st.header("Step 1: Upload Pattern HTML Files")
    pattern_files = st.file_uploader("Select HTML Pattern Files", type=['html'], accept_multiple_files=True)
    pattern_data = None
    pattern_file_path = None

    if pattern_files:
        with st.spinner("Processing pattern files..."):
            all_unique_tags = set()
            all_pattern_data = []
            fragment_id = 0
            for uploaded_file in pattern_files:
                html_content = uploaded_file.read().decode('utf-8')
                file_unique_tags = extract_unique_body_tags(html_content)
                all_unique_tags.update(file_unique_tags)
                fragment_data, fragment_id = extract_fragments(
                    html_content,
                    fragment_id,
                    uploaded_file.name.split('.')[0],
                    'body'
                )
                if fragment_data:
                    all_pattern_data.extend(fragment_data)
            global tags
            tags = sorted(list(all_unique_tags))
            columns = ["fragment_id", "fragment_name", "Level", "element_type", "element_class", "attributes", "structure"] + \
                      [f'Level_{level}_{tag}' for level in levels for tag in tags]
            df_pattern = pd.DataFrame(all_pattern_data, columns=columns)
            presence_encoding_cols = [col for col in df_pattern.columns if col.startswith("Level_")]
            df_pattern[presence_encoding_cols] = df_pattern[presence_encoding_cols].fillna(0).astype(int)
            df_pattern['Level'] = df_pattern['Level'].fillna(0).astype(int)
            df_pattern['element_type'] = df_pattern['element_type'].fillna('unknown')
            df_pattern['element_class'] = df_pattern['element_class'].fillna('')
            df_pattern['attributes'] = df_pattern['attributes'].fillna('')
            df_pattern['structure'] = df_pattern['structure'].fillna('')
            df_pattern = df_pattern[~((df_pattern['Level'] == 0) & (df_pattern['element_type'] == 'body'))]
            pattern_data = df_pattern
            st.success(f"Processed {len(pattern_files)} pattern files successfully.")
            # Save pattern data to session state for later use
            st.session_state['pattern_data'] = pattern_data

    st.header("Step 2: Upload ZIP File for Analysis")
    zip_file = st.file_uploader("Select ZIP File", type=['zip'])

    if zip_file and 'pattern_data' in st.session_state:
        if st.button("Upload and Process"):
            with st.spinner("Processing ZIP file..."):
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_zip:
                    tmp_zip.write(zip_file.read())
                    tmp_zip_path = tmp_zip.name
                combined_html, fragment_sources = extract_zip_and_combine_chapters(tmp_zip_path)
                if not combined_html:
                    st.error("Failed to extract and combine chapters from ZIP file.")
                    return
                initial_fragment_id = 0
                fragment_name = os.path.splitext(os.path.basename(tmp_zip_path))[0]
                soup = BeautifulSoup(combined_html, 'html.parser')
                initial_parent_tag = soup.find('section').name if soup.find('section') else 'body'
                fragment_data, initial_fragment_id = extract_fragments(
                    combined_html, initial_fragment_id, fragment_name, initial_parent_tag, fragment_sources
                )
                if not fragment_data:
                    st.error("No valid HTML files processed.")
                    return
                columns = ["fragment_id", "fragment_name", "Level", "element_type", "element_class",
                           "attributes", "structure", "original_file", "original_line"] + \
                          [f'Level_{lvl}_{tag}' for lvl in levels for tag in tags]
                df_combined = pd.DataFrame(fragment_data, columns=columns)
                pattern_data = st.session_state['pattern_data']
                df_combined = match_patterns(df_combined, pattern_data)
                df_clustered = perform_clustering(df_combined)
                clusters, noise = structure_clusters(df_clustered)
                report = generate_analysis_report(df_clustered, clusters, noise, pattern_data)
                st.session_state['report'] = report
                st.success("Processing completed successfully.")

    if 'report' in st.session_state:
        st.header("Fragments with Pattern Matches")
        fragments = st.session_state['report']['noise_fragment_analysis']['fragments']
        if fragments:
            df_fragments = pd.DataFrame(fragments)
            
            # Ensure all expected columns exist in df_fragments
            expected_columns = [
                "fragment_id", "fragment_name", "Level", "element_type", "element_class",
                "attributes", "structure", "original_file", "original_line"
                # Add any other columns you use if needed
            ]
            for col in expected_columns:
                if col not in df_fragments.columns:
                    df_fragments[col] = ""
            
            # --- UI-friendly pattern matches (with bullet points) ---
            def format_pattern_matches_ui(row):
                if not row.get('pattern_comparison'):
                    return None
                unique_patterns = {}
                for pattern in row['pattern_comparison']:
                    tag_structure = tuple(sorted(
                        (tag['level'], tag['tag'])
                        for tag in pattern.get('pattern_tag_structure', [])
                    ))
                    pattern_key = (pattern['pattern_name'], tag_structure)
                    if pattern_key not in unique_patterns:
                        unique_patterns[pattern_key] = {
                            'pattern_name': pattern['pattern_name'],
                            'tag_structure': tag_structure,
                            'class_mismatches': set(),
                            'missing_elements': set()
                        }
                    if pattern.get('mismatch_details') and pattern['mismatch_details'].get('class_mismatch'):
                        fragment_class = pattern['mismatch_details']['class_mismatch']['fragment_class']
                        pattern_class = pattern['mismatch_details']['class_mismatch']['pattern_class']
                        if fragment_class or pattern_class:
                            unique_patterns[pattern_key]['class_mismatches'].add(
                                f"The class '{fragment_class}' in the fragment does not match the expected class '{pattern_class}' from the pattern"
                            )
                    if pattern.get('mismatch_details') and pattern['mismatch_details'].get('tag_structure_mismatches'):
                        for mismatch in pattern['mismatch_details']['tag_structure_mismatches']:
                            unique_patterns[pattern_key]['missing_elements'].add(
                                f"The element '{mismatch['tag']}' at level {mismatch['level']} is missing"
                            )
                pattern_info = []
                for pattern_data in unique_patterns.values():
                    info = []
                    info.append(f"<b>Pattern:</b> {pattern_data['pattern_name']}")
                    if pattern_data['tag_structure']:
                        tag_structure_str = ", ".join(f"Level {level}: {tag}" for level, tag in sorted(pattern_data['tag_structure']))
                        info.append(f"<b>Tag Structure:</b> {tag_structure_str}")
                    if pattern_data['class_mismatches']:
                        for cm in sorted(pattern_data['class_mismatches']):
                            info.append(f"<b>Class Mismatch:</b> {cm}")
                    if pattern_data['missing_elements']:
                        for me in sorted(pattern_data['missing_elements']):
                            info.append(f"<b>Missing Element:</b> {me}")
                    pattern_info.append('<br>'.join(info))
                return '<hr>'.join(pattern_info) if pattern_info else None

            # --- Excel-friendly pattern matches (single-line) ---
            def format_pattern_matches_plain(row):
                if not row.get('pattern_comparison'):
                    return None
                unique_patterns = {}
                for pattern in row['pattern_comparison']:
                    tag_structure = tuple(sorted(
                        (tag['level'], tag['tag'])
                        for tag in pattern.get('pattern_tag_structure', [])
                    ))
                    pattern_key = (pattern['pattern_name'], tag_structure)
                    if pattern_key not in unique_patterns:
                        unique_patterns[pattern_key] = {
                            'pattern_name': pattern['pattern_name'],
                            'tag_structure': tag_structure,
                            'class_mismatches': set(),
                            'missing_elements': set()
                        }
                    if pattern.get('mismatch_details') and pattern['mismatch_details'].get('class_mismatch'):
                        fragment_class = pattern['mismatch_details']['class_mismatch']['fragment_class']
                        pattern_class = pattern['mismatch_details']['class_mismatch']['pattern_class']
                        if fragment_class or pattern_class:
                            unique_patterns[pattern_key]['class_mismatches'].add(
                                f"Fragment:'{fragment_class}'≠Pattern:'{pattern_class}'"
                            )
                    if pattern.get('mismatch_details') and pattern['mismatch_details'].get('tag_structure_mismatches'):
                        for mismatch in pattern['mismatch_details']['tag_structure_mismatches']:
                            unique_patterns[pattern_key]['missing_elements'].add(
                                f"{mismatch['tag']}@{mismatch['level']}"
                            )
                pattern_info = []
                for pattern_data in unique_patterns.values():
                    info = []
                    info.append(f"Pattern:{pattern_data['pattern_name']}" )
                    if pattern_data['tag_structure']:
                        tag_structure_str = "|".join(f"{tag}@{level}" for level, tag in sorted(pattern_data['tag_structure']))
                        info.append(f"Tags:{tag_structure_str}")
                    if pattern_data['class_mismatches']:
                        info.append(f"ClassMismatch:{'|'.join(sorted(pattern_data['class_mismatches']))}")
                    if pattern_data['missing_elements']:
                        info.append(f"Missing:{'|'.join(sorted(pattern_data['missing_elements']))}")
                    pattern_info.append("; ".join(info))
                return " || ".join(pattern_info) if pattern_info else None

            # --- Tag structure mismatches column ---
            def format_tag_structure_mismatches(row):
                if not row.get('pattern_comparison'):
                    return None
                mismatches = set()
                for pattern in row['pattern_comparison']:
                    if pattern.get('mismatch_details') and pattern['mismatch_details'].get('tag_structure_mismatches'):
                        for mismatch in pattern['mismatch_details']['tag_structure_mismatches']:
                            mismatches.add(f"Level {mismatch['level']}: {mismatch['tag']}")
                return ", ".join(sorted(mismatches)) if mismatches else None

            # Ensure these columns are created before any filtering or tab logic
            df_fragments['pattern_matches'] = df_fragments.apply(format_pattern_matches_ui, axis=1)
            df_fragments['pattern_matches_plain'] = df_fragments.apply(format_pattern_matches_plain, axis=1)
            df_fragments['tag_structure_mismatches'] = df_fragments.apply(format_tag_structure_mismatches, axis=1)

            # Filter out fragments with no pattern matches for some views
            df_fragments_with_matches = df_fragments[df_fragments['pattern_matches'].notna()]

            # Identify noise fragments (those with no element_class or cluster assignment)
            # We'll assume noise fragments are those with empty or missing 'element_class'
            noise_mask = df_fragments['element_class'].isna() | (df_fragments['element_class'] == '')
            noise_fragments = df_fragments[noise_mask]
            noise_fragments_with_matches = noise_fragments[noise_fragments['pattern_matches'].notna()]

            # All fragments (no filter)
            all_fragments = df_fragments.copy()

            # UI Tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "Noisy Fragments with Pattern Matches",
                "All Fragments with Pattern Matches",
                "All Noise Fragments",
                "All Fragments"
            ])

            # --- Tab 1: Only noisy fragments with matches ---
            with tab1:
                st.subheader("Noisy Fragments with Pattern Matches")
                if not noise_fragments_with_matches.empty:
                    display_df = noise_fragments_with_matches[['fragment_id', 'element_type', 'source_file', 'starting_line_number', 'pattern_matches', 'tag_structure_mismatches']]
                    st.write("<style>td {vertical-align:top !important;}</style>", unsafe_allow_html=True)
                    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    export_df = noise_fragments_with_matches[['fragment_id', 'element_type', 'source_file', 'starting_line_number', 'pattern_matches_plain', 'tag_structure_mismatches']].copy()
                    st.download_button(
                        label="Download Noisy Fragments Table (CSV)",
                        data=export_df.to_csv(index=False),
                        file_name="noisy_pattern_matches_export.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No noisy fragments with pattern matches found.")

            # --- Tab 2: All fragments with pattern matches ---
            with tab2:
                st.subheader("All Fragments with Pattern Matches")
                if not df_fragments_with_matches.empty:
                    display_df = df_fragments_with_matches[['fragment_id', 'element_type', 'source_file', 'starting_line_number', 'pattern_matches', 'tag_structure_mismatches']]
                    st.write("<style>td {vertical-align:top !important;}</style>", unsafe_allow_html=True)
                    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    export_df = df_fragments_with_matches[['fragment_id', 'element_type', 'source_file', 'starting_line_number', 'pattern_matches_plain', 'tag_structure_mismatches']].copy()
                    st.download_button(
                        label="Download All Pattern Matches Table (CSV)",
                        data=export_df.to_csv(index=False),
                        file_name="all_pattern_matches_export.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No fragments with pattern matches found.")

            # --- Tab 3: All noise fragments ---
            with tab3:
                st.subheader("All Noise Fragments")
                if not noise_fragments.empty:
                    display_df = noise_fragments[['fragment_id', 'element_type', 'source_file', 'starting_line_number', 'pattern_matches', 'tag_structure_mismatches']]
                    st.write("<style>td {vertical-align:top !important;}</style>", unsafe_allow_html=True)
                    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    export_df = noise_fragments[['fragment_id', 'element_type', 'source_file', 'starting_line_number', 'pattern_matches_plain', 'tag_structure_mismatches']].copy()
                    st.download_button(
                        label="Download All Noise Fragments Table (CSV)",
                        data=export_df.to_csv(index=False),
                        file_name="all_noise_fragments_export.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No noise fragments found.")

            # --- Tab 4: All fragments ---
            with tab4:
                st.subheader("All Fragments")
                if not all_fragments.empty:
                    display_df = all_fragments[['fragment_id', 'element_type', 'source_file', 'starting_line_number', 'pattern_matches', 'tag_structure_mismatches']]
                    st.write("<style>td {vertical-align:top !important;}</style>", unsafe_allow_html=True)
                    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    export_df = all_fragments[['fragment_id', 'element_type', 'source_file', 'starting_line_number', 'pattern_matches_plain', 'tag_structure_mismatches']].copy()
                    st.download_button(
                        label="Download All Fragments Table (CSV)",
                        data=export_df.to_csv(index=False),
                        file_name="all_fragments_export.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No fragments found.")

if __name__ == "__main__":
    main()
