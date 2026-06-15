import os
import pandas as pd

def load_raw_data():
    """
    Locates and reads the raw agent tool dataset from the structured data directory.
    Returns a pandas DataFrame.
    """
    # 1. Get the directory where this loader.py file is sitting
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Go up two levels to the main folder, then down into data/raw/
    base_dir = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(base_dir, "data", "raw", "agent_tool_tasks.csv")
    
    # 3. Check if the file is actually there before trying to open it
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"The raw data file was not found at: {data_path}\n"
            "Please double-check that 'agent_tool_tasks.csv' is placed inside 'data/raw/'."
        )
        
    print(f"Successfully located and loaded raw data from: data/raw/agent_tool_tasks.csv")
    
    # 4. Read the CSV file using pandas and return it
    return pd.read_csv(data_path)

if __name__ == "__main__":
    # This block only runs if you execute this file directly to test it
    try:
        df = load_raw_data()
        print(f"Dataset successfully loaded with {df.shape[0]} rows and {df.shape[1]} columns!")
    except Exception as e:
        print(f"❌ Error loading data: {e}")