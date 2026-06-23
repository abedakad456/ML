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

## Objective
1. **Classification:** Predict `can_answer` using K-Nearest Neighbors, AdaBoost, and additional supervised methods, specifically optimizing the trade-off between False Positives (unwanted tool calls) and False Negatives (unnecessary refusals).
2. **Clustering:** Discover latent groupings of agent-tool situations using K-Means and Agglomerative Hierarchical Clustering.

## Key Findings & Results

### Classification — Agent Aggression vs. Caution

- **Model Performance:** Tree-based ensemble models (Random Forest and AdaBoost) significantly outperformed K-Nearest Neighbors, handling severe class imbalance effectively through implicit feature selection as a natural byproduct of tree splitting.

- **Deployment Strategy:** Results strongly support a conservative deployment posture. Minimizing False Positives (Precision) was prioritized over minimizing False Negatives (Recall) — an agent that declines a borderline request is safer than one that over-triggers, leaks data to irrelevant APIs, or executes unintended actions.

### Clustering — Operational Profile Discovery

K-Means (validated via WCSS and Silhouette scores) and Agglomerative clustering (Ward linkage + Dendrograms) jointly identified **5 optimal clusters**. Cross-referencing these with Random Forest error rates revealed distinct agent-tool operational profiles:

- **The Massive Outliers** — Extremely long, noisy queries (median 1,400+ words). The model reliably identifies and rejects these, achieving 100% True Negative accuracy.

- **The Clear-Cut Math** — Well-structured, tool-aligned tasks with strong domain signals. The model correctly invokes tools ~90% of the time, yielding 0.0% False Negatives.

- **The Complex Multi-Tool Traps** — Tasks paired with a large pool of available tools (median: 5). Tool overabundance confuses the agent, inflating False Positive rates as it pattern-matches prompts to loosely relevant tools.

- **The Ambiguous Edge Cases** — Mid-length queries with no clear domain signal. The model struggles most here to reliably determine when refusal is warranted.

- **The Simple Baseline** — The dominant cluster at 56% of the dataset, representing standard, everyday operations. Queries are short (median 12 words), complexity is low, and typically only one tool is available. Operating in its most familiar territory, the model performs with low and symmetric error rates (~5.3% for both FP and FN).

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Place the raw dataset `agent_tool_tasks.csv` in `data/raw/`.
3. Proceed to `notebooks/01_exploratory_analysis.ipynb`.