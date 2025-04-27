import os
import shutil
import uuid
import json
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import numpy as np
import zipfile
import logging
import time
from bs4 import BeautifulSoup,Tag
from joblib import Parallel, delayed
from functools import lru_cache
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse import csr_matrix

logging.basicConfig(level=logging.INFO)
app = Flask(__name__,template_folder='templates')
parent_stack = []
UPLOAD_FOLDER = 'uploads'
PATTERN_UPLOAD_FOLDER = 'pattern_uploads'
CLUSTER_OUTPUT_FOLDER = 'clustered_output'
TEMP_FOLDER = 'temp'


os.makedirs(PATTERN_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLUSTER_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'zip','html'}
tags = []

levels = range(1, 14)
decorative_tags = ['em', 'strong', 'i', 'br', 'span', 'dfn']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

from bs4.element import Tag

@lru_cache(maxsize=100)
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
    global parent_stack
    parent_stack = []
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        data = []
        fragment_id = initial_fragment_id
        
        sections = soup.find_all('section')
        for idx, section in enumerate(sections):
            fragment_id += 1
            parent_tag = initial_parent_tag
            
            
            original_file = None
            original_line = None
            if fragment_sources and idx < len(fragment_sources):
                original_file = fragment_sources[idx].get('original_file')
                original_line = fragment_sources[idx].get('line_number')
            
            for child in section.children:
                if child.name and child.name not in decorative_tags:
                    parse_element(
                        child, 1, str(uuid.uuid4()), data, fragment_name, 
                        parent_tag, None, original_file, original_line
                    )
        return data, fragment_id
    
    except Exception as e:
        print(f"Error processing fragment {fragment_id}: {e}")
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
    """Original clustering function."""
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
        logging.error(f"Error in match_patterns: {str(e)}")
        df['element_class'] = None
        df['match'] = 0
        
    return df

def extract_zip_and_combine_chapters(zip_path):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_FOLDER)
        combined_html = ""
        fragment_data = []
        
        for root, dirs, files in os.walk(TEMP_FOLDER):
            if os.path.basename(root).lower().startswith('chapter'):
                for file in files:
                    if file.endswith('.html') or file.endswith('.xhtml'):
                        html_path = os.path.join(root, file)
                        with open(html_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            soup = BeautifulSoup(html_content, 'html.parser')
                            for section in soup.find_all('section'):
                                
                                fragment_info = {
                                    'html': section.prettify(),
                                    'original_file': html_path,
                                    'line_number': section.sourceline
                                }
                                fragment_data.append(fragment_info)
                                combined_html += section.prettify()
        return combined_html, fragment_data
    except Exception as e:
        print(f"Error extracting and processing ZIP file: {e}")
        return None, []

import re
from collections import defaultdict

def extract_specific_tag_classes(fragment, current_tag):
    tag_specific_classes = []

    
    for attr_segment in fragment.attributes.split(';'):
        
        if 'class' in attr_segment and current_tag in attr_segment:
            
            matches = re.findall(r'class=["\']([^"\']+)["\']', attr_segment)

            
            tag_specific_classes.extend(matches)

    return sorted(set(tag_specific_classes))


from bs4 import BeautifulSoup, Tag

def find_pattern_mismatch_reasons(fragment, pattern_data):
    if isinstance(pattern_data, list):
        pattern_data = pd.DataFrame(pattern_data)
    
    mismatch_reasons = []
    
    
    fragment_encoding = {}
    for key, value in fragment.items():
        if isinstance(key, str) and key.startswith('Level_'):
            if isinstance(value, dict):
                fragment_encoding[key] = value.get('value', 0)
            else:
                fragment_encoding[key] = value
    
    fragment_element_type = fragment.get('element_type')
    if not fragment_element_type:
        return mismatch_reasons
    
    
    matching_patterns = pattern_data[pattern_data['element_type'] == fragment_element_type]
    
    if matching_patterns.empty:
        mismatch_reasons.append({
            'pattern': 'No matching patterns found',
            'reason': f"No patterns found for element type '{fragment_element_type}'",
            'details': [],
            'mismatches': [],
            'missing_in_pattern': [],
            'missing_in_fragment': [],
            'class_mismatch': None
        })
        return mismatch_reasons
    
    
    for idx, pattern in matching_patterns.iterrows():
        pattern_name = pattern.get('fragment_name', f'Pattern #{idx}')
        pattern_element_type = pattern.get('element_type', 'Unknown')
        
        
        mismatch_data = {
            'pattern': pattern_name,
            'reason': "Structure differences found",
            'mismatches': [],
            'missing_in_pattern': [],
            'missing_in_fragment': [],
            'class_mismatch': None
        }
        
        
        for key in fragment_encoding:
            if key in pattern:
                pattern_value = pattern[key]
                fragment_value = fragment_encoding[key]
                
                if pattern_value != fragment_value:
                    level, tag = key.split('_')[1:]
                    mismatch_data['mismatches'].append({
                        'field': key,
                        'level': level,
                        'tag': tag,
                        'fragment_value': fragment_value,
                        'pattern_value': pattern_value
                    })
        
        
        for key in fragment_encoding:
            if key not in pattern and fragment_encoding[key] == 1:
                level, tag = key.split('_')[1:]
                mismatch_data['missing_in_pattern'].append({
                    'field': key,
                    'level': level,
                    'tag': tag
                })
        
        
        for key in pattern:
            if isinstance(key, str) and key.startswith('Level_') and key not in fragment_encoding and pattern[key] == 1:
                level, tag = key.split('_')[1:]
                mismatch_data['missing_in_fragment'].append({
                    'field': key,
                    'level': level,
                    'tag': tag
                })
        
        
        fragment_class = fragment.get('element_class', '')
        pattern_class = pattern.get('element_class', '')
        if fragment_class != pattern_class:
            mismatch_data['class_mismatch'] = {
                'fragment_class': fragment_class,
                'pattern_class': pattern_class
            }
        
        mismatch_reasons.append(mismatch_data)
    
    return mismatch_reasons


import json
import os
import pandas as pd

def generate_analysis_report(df_clustered, clusters, noise, output_path, pattern_data=None):
    report = {
        "overall_statistics": {},
        "cluster_statistics": {},
        "element_class_distribution": {},
        "noise_fragment_analysis": {
            "total_noise_fragments": len(noise),
            "fragments": []
        }
    }
    
    # Overall statistics
    total_fragments = len(df_clustered)
    matched_fragments = int(df_clustered['match'].sum())  # Convert to int for JSON serialization
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
    
    # Cluster statistics
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
    
    # Element class distribution
    class_distribution = df_clustered['element_class'].value_counts()
    report["element_class_distribution"] = [
        {
            "class_name": "No class assigned" if pd.isna(class_name) else class_name,
            "count": int(count)  # Convert to int for JSON serialization
        }
        for class_name, count in class_distribution.items()
    ]
    
    # Noise fragment analysis
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
                    "level": int(level),  # Convert to int for JSON serialization
                    "tag": tag
                })
        
        # Get HTML structure
        starting_line = fragment.get('original_line')
        if starting_line and starting_line != 'Unknown' and isinstance(starting_line, (int, float, str)):
            try:
                starting_line = int(float(starting_line))  # Handles numeric strings
                if pd.isna(starting_line):  # Check for NaN if it slipped through
                    starting_line = 1
            except (ValueError, TypeError):
                starting_line = 1
        else:
            starting_line = 1
        
        structure = fragment.get('structure', '')
        lines = structure.split('\n')
        html_structure = [
            f"Line {line_num}: {line.strip()}"
            for line_num, line in enumerate(lines, starting_line)
            if line.strip()
        ]
        
        # Pattern comparison analysis
        pattern_comparison = None
        if pattern_data is not None:
            filtered_patterns = pattern_data[pattern_data['element_type'] == element_type]
            mismatch_reasons = find_pattern_mismatch_reasons(fragment, filtered_patterns)
            
            pattern_comparison = []
            seen_patterns = set()
            
            for reason_data in mismatch_reasons:
                pattern_name = reason_data['pattern']
                if pattern_name in seen_patterns or pattern_name == 'No matching patterns found':
                    continue
                seen_patterns.add(pattern_name)
                
                pattern_info = {
                    "pattern_name": pattern_name,
                    "mismatch_details": {}
                }
                
                matching_pattern_rows = filtered_patterns[filtered_patterns['fragment_name'] == pattern_name]
                
                if not matching_pattern_rows.empty:
                    matching_pattern = matching_pattern_rows.iloc[0]
                    pattern_encoding_fields = []
                    for key, value in matching_pattern.items():
                        if isinstance(key, str) and key.startswith('Level_') and value == 1:
                            level, tag = key.split('_')[1:]
                            pattern_encoding_fields.append({
                                "level": int(level),  # Convert to int for JSON serialization
                                "tag": tag
                            })
                    
                    pattern_info["pattern_tag_structure"] = pattern_encoding_fields or None
                
                if reason_data.get('class_mismatch'):
                    class_info = reason_data['class_mismatch']
                    pattern_info["mismatch_details"]["class_mismatch"] = {
                        "fragment_class": class_info['fragment_class'],
                        "pattern_class": class_info['pattern_class']
                    }
                
                if reason_data.get('mismatches'):
                    mismatches = []
                    for mismatch in reason_data['mismatches']:
                        fragment_val = mismatch['fragment_value']
                        if isinstance(fragment_val, dict): fragment_val = fragment_val.get('value', 0)
                        pattern_val = mismatch['pattern_value']
                        if isinstance(pattern_val, dict): pattern_val = pattern_val.get('value', 0)
                        
                        mismatches.append({
                            "level": int(mismatch['level']),  # Convert to int for JSON serialization
                            "tag": mismatch['tag'],
                            "fragment_value": int(fragment_val),  # Convert to int for JSON serialization
                            "pattern_value": int(pattern_val)  # Convert to int for JSON serialization
                        })
                    pattern_info["mismatch_details"]["tag_structure_mismatches"] = mismatches
                
                if reason_data.get('missing_in_pattern'):
                    pattern_info["mismatch_details"]["tags_missing_in_pattern"] = [
                        {"level": int(missing['level']), "tag": missing['tag']}  # Convert to int for JSON serialization
                        for missing in reason_data['missing_in_pattern']
                    ]
                
                if reason_data.get('missing_in_fragment'):
                    pattern_info["mismatch_details"]["tags_missing_in_fragment"] = [
                        {"level": int(missing['level']), "tag": missing['tag']}  # Convert to int for JSON serialization
                        for missing in reason_data['missing_in_fragment']
                    ]
                
                pattern_comparison.append(pattern_info)
            
            if not pattern_comparison:
                pattern_comparison = {
                    "message": f"No matching patterns found for element type '{element_type}'"
                }
        
        noise_fragment = {
            "fragment_number": i+1,
            "fragment_id": fragment_id,
            "element_type": element_type,
            "source_file": original_file,
            "starting_line_number": 1 if pd.isna(original_line) or original_line in ["Nan"] else int(original_line),
            "tag_structure": encoding_fields,
            "html_structure": html_structure
        }
        
        if pattern_comparison is not None:
            noise_fragment["pattern_comparison"] = pattern_comparison
        
        report["noise_fragment_analysis"]["fragments"].append(noise_fragment)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return output_path

def extract_classes_from_structure(structure, element_type, level):
    soup = BeautifulSoup(structure, 'html.parser')
    classes = []
    
    def process_element(element, current_level):
        if current_level == level:
            if element.name == element_type:
                print(f"Found element: {element.name} with classes: {[cls.strip() for cls in element.get('class', [])]}")
                classes.extend([cls.strip() for cls in element.get('class', [])])
        elif current_level < level:
            for child in element.children:
                if isinstance(child, Tag):
                    process_element(child, current_level + 1)
    
    for child in soup.children:
        if isinstance(child, Tag):
            process_element(child, 1)
    print(f"Extracted classes: {classes}")
    return sorted(set(classes))

def structure_clusters(df):
    clusters = {}
    noise = []

    element_types = set(fragment.element_type for fragment in df.itertuples(index=False))

    for element_type in element_types:
        cluster_name = f"cluster_{element_type}"
        cluster_num = next((num for num in range(1, 100) if f"cluster{num}" not in clusters), None)

        if cluster_num is None:
            cluster_num = max(int(k.split('cluster')[-1]) for k in clusters.keys()) + 1

        cluster_data = [fragment for fragment in df.itertuples(index=False) if fragment.element_type == element_type]

        if not cluster_data:
            print(f"No data for element type {element_type}")
        else:
            cluster_groups = defaultdict(list)

            for fragment in cluster_data:
                element_class = fragment.element_class

                level_encodings = {}
                for attr in dir(fragment):
                    if attr.startswith("Level_") and getattr(fragment, attr) == 1:
                        level, tag = attr.split("_")[1:]
                        
                        classes = extract_classes_from_structure(fragment.structure, tag, int(level))

                        level_encodings[attr] = {
                            "value": 1,
                            "class": classes
                        }

                if isinstance(element_class, float) and np.isnan(element_class):
                    noise_fragment = {
                        "fragment_id": fragment.fragment_id,
                        "fragment_name": fragment.fragment_name,
                        "Level": fragment.Level,
                        "element_type": fragment.element_type,
                        "element_class": "",
                        "full_attributes": fragment.attributes,
                        "structure": fragment.structure,
                        "original_file": fragment.original_file if hasattr(fragment, 'original_file') else None,
                        "original_line": fragment.original_line if hasattr(fragment, 'original_line') else None,
                        **level_encodings
                    }
                    noise.append(noise_fragment)
                    continue

                filtered_fragment = {
                    "fragment_id": fragment.fragment_id,
                    "fragment_name": fragment.fragment_name,
                    "Level": fragment.Level,
                    "element_type": fragment.element_type,
                    "element_class": element_class or "",
                    "full_attributes": fragment.attributes,
                    "structure": fragment.structure,
                    "original_file": fragment.original_file if hasattr(fragment, 'original_file') else None,
                    "original_line": fragment.original_line if hasattr(fragment, 'original_line') else None,
                    **level_encodings
                }
                group_key=f"{element_class}" if element_class else "unknown"

                cluster_groups[group_key].append(filtered_fragment)

            if cluster_groups:
                clusters[cluster_name] = dict(cluster_groups)

    return clusters, noise

@app.route('/')
def index():
    return render_template('index.html')

def process_html_file(args):
    file_path, fragment_id, fragment_name = args
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        fragment_data, new_fragment_id = extract_fragments(html_content, fragment_id, fragment_name, None)
        return fragment_data, new_fragment_id
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {str(e)}")
        return [], fragment_id

@app.route('/upload_patterns', methods=['POST'])
def upload_patterns():
    try:
        if 'pattern_files' not in request.files:
            return jsonify({'error': 'No pattern files in the request'}), 400
        
        pattern_files = request.files.getlist('pattern_files')
        if not pattern_files or all(not file.filename for file in pattern_files):
            return jsonify({'error': 'No pattern files selected'}), 400
        
        pattern_dir = os.path.join(PATTERN_UPLOAD_FOLDER, str(uuid.uuid4()))
        os.makedirs(pattern_dir, exist_ok=True)
        
        saved_files = []
        for file in pattern_files:
            if file and file.filename.endswith('.html'):
                filename = secure_filename(file.filename)
                file_path = os.path.join(pattern_dir, filename)
                file.save(file_path)
                saved_files.append(file_path)
        
        if not saved_files:
            return jsonify({'error': 'No valid HTML pattern files uploaded'}), 400
        
        global tags
        all_unique_tags = set()
        all_pattern_data = []
        fragment_id = 0
        
        for file_path in saved_files:
            with open(file_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            file_unique_tags = extract_unique_body_tags(html_content)
            all_unique_tags.update(file_unique_tags)
            
            fragment_data, fragment_id = extract_fragments(
                html_content, 
                fragment_id, 
                os.path.splitext(os.path.basename(file_path))[0],
                'body'
            )
            
            if fragment_data:
                all_pattern_data.extend(fragment_data)
        
        tags = sorted(list(all_unique_tags))
        
        columns = ["fragment_id", "fragment_name", "Level", "element_type", "element_class", "attributes", "structure"] + \
                 [f'Level_{level}_{tag}' for level in levels for tag in tags]
        df_pattern = pd.DataFrame(all_pattern_data, columns=columns)
        
        # Fill NA values with 0 before converting to int
        presence_encoding_cols = [col for col in df_pattern.columns if col.startswith("Level_")]
        df_pattern[presence_encoding_cols] = df_pattern[presence_encoding_cols].fillna(0)
        df_pattern[presence_encoding_cols] = df_pattern[presence_encoding_cols].astype(int)
        
        # Handle NA values in other columns
        df_pattern['Level'] = df_pattern['Level'].fillna(0).astype(int)
        df_pattern['element_type'] = df_pattern['element_type'].fillna('unknown')
        df_pattern['element_class'] = df_pattern['element_class'].fillna('')
        df_pattern['attributes'] = df_pattern['attributes'].fillna('')
        df_pattern['structure'] = df_pattern['structure'].fillna('')
        
        df_pattern = df_pattern[~((df_pattern['Level'] == 0) & (df_pattern['element_type'] == 'body'))]
        
        pattern_json_path = os.path.join(PATTERN_UPLOAD_FOLDER, f"pattern_{uuid.uuid4()}.json")
        pattern_json = df_pattern.to_json(orient='records')
        with open(pattern_json_path, 'w') as json_file:
            json_file.write(pattern_json)
        
        return jsonify({
            'success': True,
            'message': f'Successfully processed {len(saved_files)} pattern files',
            'pattern_count': len(df_pattern),
            'pattern_file': pattern_json_path,
            'tags': tags,
            'pattern_files': [os.path.basename(f) for f in saved_files]
        })
        
    except Exception as e:
        logging.error(f"Error processing pattern files: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_zip_chunk(args):
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
        logging.error(f"Error processing zip chunk: {str(e)}")
        return "", []

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        if 'zip_file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400
        
        zip_file = request.files['zip_file']
        if not zip_file or not zip_file.filename.endswith('.zip'):
            return jsonify({'error': 'Invalid file format. Please upload a ZIP file.'}), 400
        
        pattern_file_path = request.form.get('pattern_file_path')
        if not pattern_file_path or not os.path.exists(pattern_file_path):
            return jsonify({'error': 'Pattern file not found. Please upload patterns first.'}), 400
        
        filename = secure_filename(zip_file.filename)
        zip_path = os.path.join(UPLOAD_FOLDER, filename)
        zip_file.save(zip_path)
        
        start_time = time.time()
        
        # Process ZIP file in chunks using multiprocessing
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.namelist()
            chunk_size = max(1, len(files) // (multiprocessing.cpu_count() * 2))
            chunks = [(zip_path, i, min(i + chunk_size, len(files))) 
                     for i in range(0, len(files), chunk_size)]
        
        combined_html = []
        fragment_sources = []
        
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() * 2) as executor:
            for html_chunk, fragment_chunk in executor.map(process_zip_chunk, chunks):
                combined_html.append(html_chunk)
                fragment_sources.extend(fragment_chunk)
        
        if not combined_html:
            return jsonify({'error': 'Failed to extract and combine chapters from ZIP file'}), 500
        
        initial_fragment_id = 0
        fragment_name = os.path.splitext(filename)[0]
        soup = BeautifulSoup(''.join(combined_html), 'html.parser')
        initial_parent_tag = soup.find('section').name if soup.find('section') else 'body'
        
        fragment_data, initial_fragment_id = extract_fragments(
            ''.join(combined_html), initial_fragment_id, fragment_name, initial_parent_tag, fragment_sources
        )
        
        if not fragment_data:
            return jsonify({'error': 'No valid HTML files processed'}), 400
        
        columns = ["fragment_id", "fragment_name", "Level", "element_type", "element_class", 
                  "attributes", "structure", "original_file", "original_line"] + \
                 [f'Level_{lvl}_{tag}' for lvl in levels for tag in tags]
        df_combined = pd.DataFrame(fragment_data, columns=columns)
        
        with open(pattern_file_path, 'r') as f:
            pattern_data = json.load(f)
        
        if isinstance(pattern_data, list):
            pattern_data = pd.DataFrame(pattern_data)
        
        df_combined = match_patterns(df_combined, pattern_data)
        df_clustered = perform_clustering(df_combined)
        
        clusters, noise = structure_clusters(df_clustered)
        output_data = {'clusters': clusters, 'noise': noise}
        
        output_filename = os.path.join(CLUSTER_OUTPUT_FOLDER, f"clustered_{filename}.json")
        with open(output_filename, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        report_filename = os.path.join(CLUSTER_OUTPUT_FOLDER, f"report_{filename}.json")
        report_path = generate_analysis_report(df_clustered, clusters, noise, report_filename, pattern_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        return jsonify({
            'success': True,
            'message': 'Processing completed successfully',
            'redirect_url': '/noise_analysis',
            'processing_time': processing_time,
            'total_fragments': len(fragment_data),
            'clustered_fragments': len(df_clustered),
            'noise_fragments': len(noise)
        })
            
    except Exception as e:
        logging.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/noise_analysis')
def noise_analysis():
    return render_template('noise_analysis.html')

@app.route('/get_report_data')
def get_report_data():
    try:
        # Get the most recent report file
        report_files = [f for f in os.listdir(CLUSTER_OUTPUT_FOLDER) if f.startswith('report_')]
        if not report_files:
            return jsonify({'error': 'No report file found'}), 404
            
        latest_report = max(report_files, key=lambda x: os.path.getctime(os.path.join(CLUSTER_OUTPUT_FOLDER, x)))
        report_path = os.path.join(CLUSTER_OUTPUT_FOLDER, latest_report)
        
        with open(report_path, 'r') as f:
            report_data = json.load(f)
            
        return jsonify(report_data)
    except Exception as e:
        logging.error(f"Error loading report data: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)