# Quality of Wikipedia Articles

This project aims to analyze and classify Wikipedia articles into promotional and non-promotional categories.

## Table of Contents

- [Installation](#installation)
- [Code Formatter](#code-formatter)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Logging](#logging)

## Installation

### Prerequisites

- Python 3.x

To install the required packages, run the following command:

```sh
pip install -r requirements.txt
```

## Code Formatter

We use Ruff for code formatting. To format the code, run:

```sh
ruff check .
```

## Usage

To run the script, use the following command:

```sh
python main.py -c <config-name>
```

- -c, --config: Name of the YAML configuration file (without the .yaml extension).

## Configuration

Specify the configuration settings in the YAML files located in the configs/ directory. It should contain the following structure:

```yaml
data_loader:
  good_file: "data/raw/good.csv" # Path to the CSV file containing non-promotional text data.
  promo_file: "data/raw/promotional.csv" # Path to the CSV file containing promotional text data.
  nrows: 1000 # (optional) Number of rows to read from each CSV file.
  shuffle: true # Whether to shuffle the combined dataset.
```

## Project Structure

- Praktikumsbericht/: Contains the LaTeX code for the report.
- configs/: Contains the YAML configuration files.
- src/: Contains the source code.
  - data_loader.py: Functions for loading and processing data.
  - utils.py: Utility functions including argument parsing and file validation.
- main.py: Main script to run the project.

## Logging

Logs are saved to app.log and also printed to the console.

## Convert Wikipedia Dump

- Download from https://dumps.wikimedia.org/enwiki/:
  - `enwiki-yyyymmdd-pages-articles-multistream.xml.bz2` (about 20 GB)
  - `enwiki-yyyymmdd-pages-articles-multistream-index.txt.bz2` (about 250 MB)
- Run `main_wp_dump_convert.py` to split the dump into 4 CSV files:
  - `wp/good.csv`
  - `wp/promotional.csv`
  - `wp/neutral.csv`
  - `wp/skipped.csv`
- Run `main_wp_dump_sampling.py` to extract samples with an equal number of entries:
  - `wp/good_sample.csv`
  - `wp/promotional_sample.csv`
  - `wp/neutral_sample.csv`
- **TODO** arguments/config to pass file names
