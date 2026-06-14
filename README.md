# Agent Tool Decision Engine

## Overview
AI agents are increasingly equipped with tools (APIs, databases, math functions). However, a critical decision an agent must make is whether to use an available tool or refuse the request. Wrong tool calls can result in wasted compute, leaked private data, or unwanted actions. 

This project tackles this binary classification task (`can_answer`), evaluating whether at least one tool matches the user's query (1) or if the agent should explicitly refuse (0).

## Dataset
The data is derived from the Berkeley Function Calling Leaderboard (BFCL) by the Gorilla project at UC Berkeley. 
- **Rows**: 3,491 agent tasks
- **Features**: Query properties, tool parameters, match dynamics, and complexity markers.
- **Target**: `can_answer` (Binary)

## Project Structure
- `notebooks/`: Jupyter notebooks for EDA, preprocessing, classification, and clustering.
- `src/`: Modularized Python scripts for data handling, feature engineering, modeling, and evaluation.
- `docs/`: Data dictionaries and methodology explanations.

## Objective
1. **Classification:** Predict `can_answer` using K-Nearest Neighbors, AdaBoost, and additional supervised methods, specifically optimizing the trade-off between False Positives (unwanted tool calls) and False Negatives (unnecessary refusals).
2. **Clustering:** Discover latent groupings of agent-tool situations using K-Means and Agglomerative Hierarchical Clustering.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Place the raw dataset `agent_tool_tasks.csv` in `data/raw/`.
3. Proceed to `notebooks/01_exploratory_analysis.ipynb`.