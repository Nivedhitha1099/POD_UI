import os
import json
import pandas as pd
from bs4 import BeautifulSoup, Tag
import numpy as np
from collections import defaultdict
import re

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

def extract_fragments(html_content, initial_fragment_id, fragment_name, initial_parent_tag, fragment_sources=None):
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
                if child.name and child.name not in ['em', 'strong', 'i', 'br', 'span', 'dfn']:
                    parse_element(
                        child, 1, str(fragment_id), data, fragment_name, 
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

    presence_encoding = {f'Level_{lvl}_{tag}': 0 for lvl in range(1, 14) for tag in ['h1', 'h2', 'h3', 'p', 'div', 'span']}
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
        if child.name and child.name not in ['em', 'strong', 'i', 'br', 'span', 'dfn']:
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
        if child.name and child.name not in ['em', 'strong', 'i', 'br', 'span', 'dfn']:
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
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.8, min_samples=2, metric='cosine')
    df['cluster'] = dbscan.fit_predict(X)
    return df

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
        print(f"Error in match_patterns: {str(e)}")
        df['element_class'] = None
        df['match'] = 0
        
    return df

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
                        level_encodings[attr] = {
                            "value": 1,
                            "class": []
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

def generate_analysis_report(df_clustered, clusters, noise, output_path, pattern_data=None):
    report = {
        "overall_statistics": {
            "total_fragments": len(df_clustered),
            "matched_fragments": {
                "count": len(df_clustered[df_clustered['match'] == 1]),
                "percentage": round(len(df_clustered[df_clustered['match'] == 1]) / len(df_clustered) * 100, 2)
            },
            "unmatched_fragments": {
                "count": len(df_clustered[df_clustered['match'] == 0]),
                "percentage": round(len(df_clustered[df_clustered['match'] == 0]) / len(df_clustered) * 100, 2)
            }
        },
        "noise_fragment_analysis": {
            "fragments": []
        }
    }
    
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
            if not filtered_patterns.empty:
                pattern_comparison = []
                for _, pattern in filtered_patterns.iterrows():
                    pattern_info = {
                        "pattern_name": pattern.get('fragment_name', 'Unknown'),
                        "pattern_tag_structure": [],
                        "mismatch_details": {}
                    }
                    
                    # Get pattern tag structure
                    for key, value in pattern.items():
                        if isinstance(key, str) and key.startswith('Level_') and value == 1:
                            level, tag = key.split('_')[1:]
                            pattern_info["pattern_tag_structure"].append({
                                "level": int(level),
                                "tag": tag
                            })
                    
                    pattern_comparison.append(pattern_info)
        
        report["noise_fragment_analysis"]["fragments"].append({
            "fragment_number": i + 1,
            "fragment_id": fragment_id,
            "fragment_name": fragment_name,
            "element_type": element_type,
            "source_file": original_file,
            "starting_line_number": starting_line,
            "tag_structure": encoding_fields,
            "html_structure": html_structure,
            "pattern_comparison": pattern_comparison
        })
    
    # Write report to file
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return output_path 