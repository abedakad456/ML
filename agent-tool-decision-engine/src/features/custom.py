import pandas as pd
import numpy as np
import re

def _get_query_text(text):
    """Helper to ensure the text is a valid, lowercase string for regex matching."""
    text = str(text).lower()
    if text == 'nan' or not text.strip():
        return ""
    return text

def calculate_comparison_keywords(text):
    """1 if query contains comparison keywords, else 0."""
    text = _get_query_text(text)
    if not text:
        return 0
    # \b ensures we match whole words only (e.g., 'than' instead of 'thank')
    pattern = r'\b(vs|compare|better|difference|versus|than)\b'
    return 1 if re.search(pattern, text) else 0

def calculate_example_keywords(text):
    """1 if query contains example specification keywords, else 0."""
    text = _get_query_text(text)
    if not text:
        return 0
    pattern = r'\b(example|like|such as|instance)\b'
    return 1 if re.search(pattern, text) else 0

def calculate_generative_query(text):
    """1 if query contains generation or creation keywords, else 0."""
    text = _get_query_text(text)
    if not text:
        return 0
    pattern = r'\b(generate|create|make|build|write|draw|design)\b'
    return 1 if re.search(pattern, text) else 0

def build_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all custom feature calculations to the dataframe."""
    df_out = df.copy()
    
    # 1. Apply column-based NLP logic directly to the query
    df_out['comparison_keywords'] = df_out['query'].apply(calculate_comparison_keywords)
    df_out['example_keywords'] = df_out['query'].apply(calculate_example_keywords)
    df_out['is_generative_query'] = df_out['query'].apply(calculate_generative_query)
    
    # 2. Calculate Total Relations
    # We safely extract the original intent score and fill NaNs with 0 just in case
    intent_score = df_out.get('query_multi_intent_score', 0).fillna(0)
    df_out['total_relations'] = (
        intent_score + 
        df_out['comparison_keywords'] + 
        df_out['example_keywords']
    )
    
    # 3. Calculate Relation Words Ratio
    # We use np.where to prevent division by zero errors if a query has 0 words
    word_count = df_out.get('query_word_count', 0).fillna(0)
    df_out['relation_words_ratio'] = np.where(
        word_count == 0, 
        0.0, 
        df_out['total_relations'] / word_count
    )
    
    return df_out