import pandas as pd
import numpy as np
import re

def fill_missing_text(df: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values in text columns before feature engineering."""
    df_out = df.copy()
    # If the AI didn't need a tool, tool_names might be NaN.
    df_out['tool_names'] = df_out['tool_names'].fillna("")
    df_out['query'] = df_out['query'].fillna("")
    return df_out

def calculate_required_params_ratio(row):
    """Formula: total_required_params / total_params. If total_params == 0, set to 0."""
    if pd.isna(row.get('total_params')) or row.get('total_params', 0) == 0:
        return 0.0
    return row['total_required_params'] / row['total_params']

def calculate_avg_params_per_tool(row):
    """Formula: total_params / num_available_tools. If num_available_tools == 0, set to 0."""
    if pd.isna(row.get('num_available_tools')) or row.get('num_available_tools', 0) == 0:
        return 0.0
    return row['total_params'] / row['num_available_tools']

def calculate_query_avg_word_length(row):
    """Average length of the words in the raw query."""
    query = str(row.get('query', ''))
    if query == 'nan' or not query.strip():
        return 0.0
    words = query.split()
    return sum(len(word) for word in words) / len(words)

def calculate_query_mentions_number(row):
    """1 if the raw query contains at least one digit, otherwise 0."""
    query = str(row.get('query', ''))
    if query == 'nan' or not query.strip():
        return 0
    return 1 if re.search(r'\d', query) else 0

def calculate_tool_name_diversity(row):
    """Number of unique tool-name prefixes before the first dot."""
    tools_str = str(row.get('tool_names', ''))
    if not tools_str or tools_str == 'nan':
        return 0
    
    # Tool names are pipe-separated
    tools = tools_str.split('|')
    prefixes = set()
    for tool in tools:
        if '.' in tool:
            prefixes.add(tool.split('.')[0])
        else:
            prefixes.add(tool) # Treating the whole name as prefix if no dot exists
            
    return len(prefixes)

def build_mandatory_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all mandatory feature calculations to the dataframe."""
    df_out = df.copy()
    
    df_out['required_params_ratio'] = df_out.apply(calculate_required_params_ratio, axis=1)
    df_out['avg_params_per_tool'] = df_out.apply(calculate_avg_params_per_tool, axis=1)
    df_out['query_avg_word_length'] = df_out.apply(calculate_query_avg_word_length, axis=1)
    df_out['query_mentions_number'] = df_out.apply(calculate_query_mentions_number, axis=1)
    df_out['tool_name_diversity'] = df_out.apply(calculate_tool_name_diversity, axis=1)
    
    return df_out