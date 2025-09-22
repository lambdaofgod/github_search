# GitHub Search - Machine Learning for Python Repository Search

## Overview

This research project implements machine learning approaches for searching and analyzing GitHub Python repositories. It uses a pipeline-based architecture with Dagster for orchestrating data processing, model training, and evaluation workflows.

## Core Components

### `github_search/dependency_records/` Module

The dependency records module provides tools for analyzing Python code structure and relationships:

- **GraphExtractor class**: The main class for dependency analysis
  - `extract_repo_dependencies_df()`: Analyzes Python code to extract comprehensive dependency graphs that capture relationships between functions, classes, imports, and method calls within repositories

### `github_search/lms/` Module

The LMS (Language Model Services) module provides utilities for code documentation generation:

- **Code2Documentation class**: Generates README documentation from repository code using a two-step hierarchical process:
  1. First summarizes individual files
  2. Then combines file summaries to create repository-level documentation

- **Code2DocumentationFlat class**: A simpler variant that directly generates documentation from concatenated code files without the intermediate summarization step

- **librarian.py**: Contains utilities for few-shot learning and prompt engineering, used by the `librarian_tasks` asset to generate task descriptions for repositories based on their dependency signatures

## Dagster Pipeline Architecture

### Data Flow

The pipeline follows this high-level data flow:

```
inputs → call_graph → code2doc/librarian → evaluation
      ↘ corpus ↗
```

### Key Assets

- **inputs**: Load and prepare repository metadata and Python source code data
- **call_graph**: Extract dependency graphs, calculate centrality metrics, and generate function signatures
- **librarian_tasks**: Generate task descriptions for repositories using few-shot learning with dependency signatures (leverages utilities from librarian.py)
- **code2doc**: Generate README documentation using language models with various strategies (flat, hierarchical, with/without dependency context)
- **corpus**: Prepare information retrieval datasets for evaluation
- **evaluation**: Calculate metrics for generated documentation quality

### Asset Groups

Assets are organized into logical groups:

- `inputs`: Data ingestion and preparation
- `call_graph`: Dependency analysis and graph extraction
- `code2doc`: Documentation generation
- `librarian`: Document expansion and task generation
- `corpus`: IR dataset preparation
- `evaluation`: Metrics and evaluation

## Installation

### Prerequisites

- Python ≥ 3.10
- CUDA GPU (required for Graph Neural Network training)
- 64GB+ RAM (recommended for large dataset processing)

### Setup

```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -e .
```

## Usage

### Starting the Dagster UI

```bash
uv run dagster-webserver -m github_search.pipelines.dagster.definitions
```

The Dagster UI will be available at http://localhost:3000

### Materializing Assets

Materialize a single asset:
```bash
uv run dagster asset materialize -m github_search.pipelines.dagster.definitions --select <asset_name>
```

Materialize multiple assets:
```bash
uv run dagster asset materialize -m github_search.pipelines.dagster.definitions --select "asset1,asset2"
```

### Example Workflows

Generate README documentation:
```bash
# Generate flat structure READMEs
uv run dagster asset materialize -m github_search.pipelines.dagster.definitions --select flat_generated_readmes

# Generate hierarchical READMEs with dependency context
uv run dagster asset materialize -m github_search.pipelines.dagster.definitions --select code2doc_readmes_from_dependency_signatures
```

Run dependency analysis:
```bash
# Extract dependency graphs
uv run dagster asset materialize -m github_search.pipelines.dagster.definitions --select graph_dependencies_df

# Generate librarian tasks
uv run dagster asset materialize -m github_search.pipelines.dagster.definitions --select librarian_tasks
```

## Configuration

The pipeline is configured using `env.yaml` which contains:
- Model parameters
- Data paths
- Processing options
- Hyperparameters

Key configuration sections:
- `code2doc`: LLM model settings for documentation generation
- `librarian`: Few-shot learning parameters
- `evaluation`: Metrics and evaluation settings

## Project Structure

```
github_search/
├── pipelines/
│   └── dagster/          # Dagster pipeline definitions
│       ├── definitions.py # Main Dagster definitions
│       ├── input_assets.py
│       ├── call_graph_assets.py
│       ├── code2doc_assets.py
│       ├── document_expansion.py
│       ├── corpora_assets.py
│       └── evaluation_assets.py
├── dependency_records/   # Code analysis tools
│   ├── python_call_graph.py
│   └── python_code_analysis.py
├── lms/                  # Language model services
│   ├── code2documentation.py
│   ├── librarian.py
│   └── repo_file_summary_provider.py
└── evaluation/           # Evaluation metrics
```

## Development

For development and debugging:

1. Use the Dagster UI to visualize asset dependencies and monitor execution
2. Check logs in the Dagster UI for detailed execution information
3. Use `--select` to test individual assets before running full pipelines

## License

This is a research project for academic purposes.