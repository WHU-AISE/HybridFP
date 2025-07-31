# HybridFP: Divide-and-conquer Serverless Function Provision for Mitigating Cold Starts

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📖 Introduction

Function as a Service (FaaS) has emerged as a prevalent cloud computing paradigm, favored by its ease of deployment, rapid elasticity, and pay-as-you-go billing. However, on-demand resource management of FaaS often introduces the cold start problem, which can increase waiting time and degrade performance. Due to the heterogeneous and dynamic nature of large-scale serverless functions, existing solutions struggle to holistically optimize the trade-off between cold start reduction and resource waste. Inspired by the divide-and-conquer idea, this paper introduces HybridFP, a hybrid function provision method designed to effectively and economically mitigate cold starts. We first partition serverless functions into model-worthy and model-unworthy functions in view of prediction profit. Subsequently, HybridFP integrates an invocation predictor and a pattern matcher to guide function provision, enabling timely function prewarming and instance eviction. Experimental results conducted on two industrial datasets demonstrate that HybridFP achieves performance on par with the state-of-the-art in mitigating cold starts, while reducing memory waste by 78.29\% and 88.6\%, respectively.


## 🏗️ Project Structure

```
HybridFP/
├── data.py                 # Data processing and feature extraction
├── common.py               # Common utility classes and functions
├── methods/                # Prediction algorithm implementations
│   ├── HybridFP.py        # Hybrid prediction algorithm (main method)
│   ├── SPES.py            # SPES prediction algorithm
│   ├── IceBreaker.py      # IceBreaker prediction algorithm
│   ├── HIST.py            # HIST prediction algorithm
│   ├── OpenWhisk.py       # OpenWhisk baseline
│   └── pattern_discovery.py # Pattern discovery algorithm
├── azure-data/            # Azure Functions dataset
│   ├── invocations_per_function_md.anon.d*.csv  # Function invocation data
│   ├── function_durations_percentiles.anon.d*.csv # Execution time data
│   ├── app_memory_percentiles.anon.d*.csv      # Memory allocation data
│   └── readme.md          # Dataset documentation
├── mid_data/              # Intermediate processed data
└── result/                # Experimental results output
```

## 🚀 Quick Start

### Requirements

- Python 3.8+
- Recommended to use conda or venv for virtual environment

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd HybridFP

# Install dependencies
pip install numpy pandas polars tqdm scipy statsmodels
pip install lightgbm LazyProphet stumpy mne
```

### Running Examples

```bash
# Run HybridFP algorithm
python methods/HybridFP.py

# Run other baseline methods
python methods/SPES.py
python methods/IceBreaker.py
python methods/HIST.py
```


## 🔬 Algorithm Methods

### HybridFP (Main Method)
- **Prediction Strategy**: Combines time series prediction and pattern matching
- **Prewarming Strategy**: Dynamically adjusts container numbers based on prediction results
- **Applicable Scenarios**: Uses LightGBM for predictable functions, pattern matching for others

### SPES (Baseline Method)
- **Prediction Strategy**: Based on function invocation pattern classification and prediction
- **Classification Types**: Regular, Appro-regular, Dense, Successive, etc.
- **Features**: Adopts different prediction strategies for different invocation patterns

### IceBreaker (Baseline Method)
- **Prediction Strategy**: Time series extrapolation based on Fourier transform
- **Applicable Scenarios**: Function invocations with periodic characteristics

### HIST (Baseline Method)
- **Prediction Strategy**: Prediction based on Inter-arrival Time (IT) histogram
- **Prewarming Strategy**: Adaptive prewarming and keep-alive time adjustment



---

**Note**: This project is for academic research purposes only. Please comply with the relevant dataset usage agreements. 