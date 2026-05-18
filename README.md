# Case Study: eyeNet — Lens GRN Explorer

![MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-4.0-informational.svg)
![NetworkX](https://img.shields.io/badge/NetworkX-3.6-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-3.0-150458.svg)
![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7.svg)

## Overview

eyeNet is an interactive web application for exploring Gene Regulatory Networks (GRNs) in lens development. Researchers can visualize, filter, and analyze regulatory interactions between transcription factors and target genes across developmental stages and tissue types.

The tool integrates microarray and RNA-seq expression data alongside curated GRN edge data, enabling both network-level and gene-level exploration through an intuitive browser-based interface.

**Stakeholders:** Computational biologists, lens development researchers, and bioinformaticians working with the Lachke lab GRN dataset.

## Features

- Interactive network graph with drag, zoom, and filter controls
- Filter by regulator, target gene, tissue type, and developmental stage
- Split network view when regulator and target tissues differ
- RNA-seq and microarray expression data integrated into a single data panel
- Master regulator and highly regulated gene identification
- Automated BioGRID data fetching to keep interaction data up to date

## Repository Structure

```
eyeNet/
├── app.py                   # Main Dash application entry point
├── grn_network.py           # Network graph logic and data loading
├── requirements.txt         # Python dependencies
├── Procfile                 # Deployment entry point
├── data/
│   ├── Lens_GRN_June_2016_original FOR HACKATHON - Salil Lachke.xlsx  # GRN edge data
│   ├── MegaTable April 24 2024 for Microarray and RNA Seq Sent to Murali (1).xls  # Microarray + RNA-seq enrichment
│   ├── LogCPM_FPKM_TPM_FiberEpi120225 USE THIS FEB 13 2025.xls        # FPKM expression values
│   ├── dataDict.json        # Data dictionary
│   └── biogridScrape/       # Downloaded BioGRID interaction files
├── src/
│   └── biogridScrape.py     # Script to fetch latest BioGRID release
├── lib/                     # Frontend JS/CSS libraries (vis.js, tom-select)
├── docs/                    # Sphinx documentation scaffold
└── .github/workflows/       # CI configuration
```

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/NSF-DARSE/eyeNet.git
cd eyeNet
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

Then open your browser to `http://127.0.0.1:8050`

## Data

The app uses three data files in the `data/` directory:

| File | Description |
|---|---|
| `Lens_GRN_June_2016_original...xlsx` | Curated GRN edge list (regulators → targets) from Lachke lab |
| `MegaTable April 24 2024...xls` | Microarray raw counts + RNA-seq enrichment data |
| `LogCPM_FPKM_TPM_FiberEpi...xls` | Log CPM / FPKM / TPM expression values (Feb 2025) |

## BioGRID Scraping

The `src/biogridScrape.py` script automatically checks for and downloads the latest BioGRID protein interaction release.

It compares the local version against the current BioGRID REST service version and only downloads if an update is available, saving the result to `data/biogridScrape/`.

### Usage

```bash
python src/biogridScrape.py
```

> **Note:** The script uses a BioGRID REST API access key. Store it in a `.env` file and never commit it directly in the code.

## Documentation

This repository includes a Sphinx documentation scaffold under `docs/`. To build the docs:

```bash
cd docs
pip install -r requirements.txt
make html
```

## Contributing

All changes must go through pull requests against `main`.

1. Clone the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes and commit (`git commit -m "feat: description"`)
4. Push to your branch (`git push origin feature/your-feature`)
5. Open a pull request early for visibility

## License

MIT — Copyright (c) 2026 NSF-DARSE. Authors: Jahnavi Gangishetti, Julia Zimmerman
