# 🌾 Coupled CH4MOD & Microbiome Digital Twin Simulator

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%205%2B-orange.svg)](https://www.gradio.app/)
[![PM4Py](https://img.shields.io/badge/Process%20Mining-PM4Py-green.svg)](https://pm4py.fit.fraunhofer.de/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces%20Compatible-yellow)](https://huggingface.co/spaces)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-red.svg)](https://www.gnu.org/licenses/agpl-3.0)

An integrated, multi-scale biogeochemical modeling framework that couples the macroscopic **CH4MOD** biophysical rice-soil model with a microscopic, **stoichiometric Petri Net** parameterised by longitudinal metagenomic correlation metrics ($r$, $p$, $p_{\text{adj}}$) across phenological stages (Weeks 3, 6, and 10 post-transplant). https://huggingface.co/spaces/TemasekLifesciLab/CH4MicrobiomeSim

---

## 📄 License
This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** due to its dependency on the open-source community edition of [PM4Py](https://pm4py.fit.fraunhofer.de/). Commercial licensing options are available through PM4Py.

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Architecture](#-model-architecture)
- [Microbial Consortium & Metagenomic Data](#-microbial-consortium--metagenomic-data)
- [Mathematical Framework](#-mathematical-framework)
- [Installation & Quickstart](#-installation--quickstart)
- [Input Data Formatting](#-input-data-formatting)
- [Web Interface (Gradio)](#-web-interface-gradio)
- [Troubleshooting & Deployment Notes](#-troubleshooting--deployment-notes)
- [Citation](#-citation)

---

## 🔬 Overview

Methane ($\text{CH}_4$) and nitrous oxide ($\text{N}_2\text{O}$) emissions from rice paddies are driven by non-linear interactions between agricultural management, soil physicochemical status ($E_h$, water table, sulfate), and microbial community succession. 

Traditional biogeochemical models treat microbial processes as generic, lumped first-order kinetic pools. This software bridges the macro-micro gap:
1. **Macroscopic Layer (CH4MOD):** Simulates daily plant biomass development, soil carbon depolymerisation, continuous redox potential ($E_h$), and physical emission pathways (plant-mediated aerenchyma transport, ebullition, diffusion) across a continuous **2-year timeline ($730\text{ days}$)**.
2. **Microscopic Layer (Petri Net):** Implements a discrete, stochastic Petri Net (using PM4Py) representing a **10-taxon syntrophic consortium**. Transition arc weights are dynamically modulated by physical redox feasibility ($R_{\text{anaer}}$), competitive sulfate inhibition ($S_{\text{sulf-supp}}$), and stage-specific metagenomic Pearson correlation coefficients ($w_i$).

```
[ Climate Data: T2M, TS, Water Depth ]
                 │
                 ▼
     ┌───────────────────────┐
     │  CH4MOD Macro-Model   │──► Daily Biomass, Redox (Eh), Daily CH4, Aerenchyma Flux
     └───────────────────────┘
                 │ (Hourly Coupling via Eh, Sulfate, Substrates)
                 ▼
     ┌───────────────────────┐
     │ Petri Net Micro-Model │──► 10 Microbial Guild Trajectories, N2O/CH4 Partitioning,
     └───────────────────────┘    Dynamic Week 3/6/10 Metagenomic Arc Weights
```

---

## ✨ Key Features

* **Continuous 2-Year Horizon ($730\text{ Days}$):** Full support for cross-year winter/late-season plantings (e.g., November–December transplanting harvesting in March–April of Year 2).
* **Empirical Metagenomic Constraint:** Dynamic arc weights driven by real amplicon sequencing across **Weeks 3, 6, and 10** post-transplantation.
* **10-Taxon Syntrophic Ecosystem:**
  * **5 Methanogens:** *Methanothrix* (acetoclastic), *Methanosarcina* (mixotrophic), *Methanocella* (Rice Cluster I), *Methanoregula* (reproductive bloom specialist), *Methanobacterium* (hydrogenotrophic).
  * **4 Syntrophic & Cycling Partners:** *Syntrophus* (fatty acid fermenter), *Desulfomonile* (sulfate-reducing competitor), *Sulfuricaulis* (sulfur oxidizer), *Nitrospira* (nitrifier / $\text{N}_2\text{O}$ precursor generator).
  * **1 Methanotroph:** *Methylocaldum* (aerobic methane oxidizer).
* **Thermodynamic Substrate Competition:** Michaelis–Menten-derived sulfate suppression governing acetotrophic methanogenesis.
* **Interactive UI:** Parameter sensitivity analysis, weather data upload, mid-season top-dressing injection events, hourly token distribution plots, and final Petri Net visual marking exports.

---

## 🧬 Microbial Consortium & Metagenomic Data

The Petri net represents 10 functional microbial taxa categorized into three initial baseline tiers:

| Taxon | Initial Tokens ($M_0$) | Guild / Functional Role | Statistical Status across Growth Stages |
| :--- | :---: | :--- | :--- |
| **`Methanothrix`** | $5{,}000$ | Obligate acetoclastic methanogenesis | Significant in Wk 3 (r=0.617***) & Wk 6 (r=0.555***) |
| **`Methanosarcina`** | $5{,}000$ | Mixotrophic methanogenesis | Significant in Wk 3 (r=0.367*) & Wk 10 (N2O=-0.297*) |
| **`Nitrospira`** | $5{,}000$ | Aerobic nitrification ($\text{NH}_4^+ \to \text{NO}_3^-$) | Significant in Wk 3 (CH4=-0.390, N2O=0.357) |
| **`Methanocella`** | $1{,}000$ | Hydrogenotrophic (Rice Cluster I) | Significant in Wk 3 (r=0.530***) & Wk 6 (r=0.405***) |
| **`Methylocaldum`** | $500$ | Aerobic methanotrophy ($\text{CH}_4 \to \text{CO}_2$) | Significant in Wk 3 (r=0.341*), Wk 6 (r=0.345**), Wk 10 (N2O=-0.347**) |
| **`Desulfomonile`** | $200$ | Sulfate-reducing bacterium (SRB) | Significant in Wk 3 (r=0.501***) & Wk 6 (r=0.442***) |
| **`Sulfuricaulis`** | $200$ | Chemolithotrophic sulfur oxidation | Ambient cycling partner ($ns$ across weeks) |
| **`Methanobacterium`** | $100$ | Hydrogenotrophic seed-bank | Significant in Wk 3 (r=0.514***) & Wk 10 (N2O=-0.371**) |
| **`Methanoregula`** | $100$ | Hydrogenotrophic seeder (Wk 6 bloom) | Significant in Wk 3 (r=0.501***) & Wk 6 surge (r=0.621***) |
| **`Syntrophus`** | $100$ | Syntrophic VFA oxidizer | Significant in Wk 6 surge (r=0.562***) |

---

## 📐 Mathematical Framework

### Dynamic Biomass Synthesis Arc Weight Equation

The token synthesis weight $W_{T \to P}(t)$ for a microbial transition $T$ firing into place $P$ is calculated hourly:

$$W_{T \to P}(t) = \max\left(1, \, \Big\lfloor W_{\text{base}} \cdot \Big(\beta + \mathcal{E}(t) \cdot \mathcal{T}(t) \cdot \mathcal{M}(t) \cdot \gamma\Big) \Big\rfloor\right)$$

#### 1. Environmental Redox Gating $\mathcal{E}(t) = R_{\text{anaer}}$
Derived from simulated continuous soil redox potential ($E_h$) via a Boltzmann logistic function:

$$R_{\text{anaer}}(t) = \frac{1}{1 + \exp\left(\frac{E_h(t) - (-100\text{ mV})}{30\text{ mV}}\right)}$$

Under aerobic conditions ($E_h \gg -100\text{ mV}$), $R_{\text{anaer}} \to 0$; under sustained flooding ($E_h \ll -100\text{ mV}$), $R_{\text{anaer}} \to 1.0$.

#### 2. Thermodynamic Sulfate Suppression $\mathcal{T}(t) = S_{\text{sulf-supp}}$
Captures competitive exclusion of acetoclastic methanogens by sulfate reducers (*Desulfomonile*):

$$S_{\text{sulf-supp}}(t) = 1.0 - 0.6 \times \left(\frac{[\text{Sulphate}]}{[\text{Sulphate}] + 1000}\right)$$

*When sulfate is abundant, acetoclastic methanogenesis is suppressed by up to 60%.*

#### 3. Metagenomic Allocation $\mathcal{M}(t) = w_i$
Piecewise continuous stage-specific weighting normalized across active guild members:

$$w_i(t) = \frac{r_i(t)}{\sum_{k \in \text{guild}} r_k(t)}$$

#### 4. Basal Maintenance Floor ($\beta = 0.05$) & Kinetic Gain ($\gamma$)
* **$\beta = 0.05$:** Allocates a $5\%$ endogenous maintenance turnover floor to preserve dormant seed-bank populations during dry/fallow periods, preventing Petri Net deadlock.
* **$\gamma$:** A kinetic amplification multiplier ($\gamma = 3.0$ for opportunistic seed-bank bloomers like *Methanoregula*; $\gamma = 2.0 - 2.5$ for dominant residents).

---

## 🚀 Installation & Quickstart

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Graphviz installed on your system (required by PM4Py for Petri Net rendering):
  * **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install -y graphviz`
  * **macOS (Homebrew):** `brew install graphviz`
  * **Windows (Chocolatey):** `choco install graphviz`

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kjxlau/CH4MicrobiomeSim.git
   cd CH4MicrobiomeSim
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the simulation app:**
   ```bash
   python app.py
   ```
   Open your browser at `http://127.0.0.1:7860`.

---

## 📄 Input Data Formatting

The application accepts NASA POWER daily reanalysis files or custom weather data covering **730 continuous days (2 years)**.

### NASA POWER Parameter Mapping
* **Air Temperature (`tair.txt`):** Parameter **`T2M`** (Temperature at 2 Meters, °C).
* **Soil Temperature (`tsoil.txt`):** Parameter **`TS`** (Earth Skin Temperature, °C).
* **Water Depth (`wd.txt`):** Daily water depth (cm, where positive values indicate floodwater depth and negative values indicate water table depth below soil surface).

### Formatting Helper (Python)
If your observed data spans only part of the 2-year timeline (e.g., Nov 1 to Jan 31), format it into 730 rows with unmonitored fallow days padded with `0.0`:

```python
import pandas as pd

# Load observed data: YEAR, DOY, TS, T2M
df = pd.read_csv("nasa_power.txt", sep=r"\s+")
df["Julian_Day"] = df.apply(lambda r: int(r["DOY"]) if r["YEAR"] == 2025 else int(r["DOY"]) + 365, axis=1)

# Merge onto 1..730 grid
full_grid = pd.DataFrame({"Julian_Day": range(1, 731)})
final_df = full_grid.merge(df[["Julian_Day", "TS", "T2M"]], on="Julian_Day", how="left").fillna(0.0)

# Export single-column files for upload
final_df["TS"].to_csv("tsoil.txt", index=False, header=False)
final_df["T2M"].to_csv("tair.txt", index=False, header=False)
```

---

## 🖥️ Web Interface (Gradio)

The application provides an interactive control panel organized into functional tabs:

1. **Crop Timelines:** Configure planting, maximum tillering, and harvest dates for up to three consecutive cropping seasons across Days 1–730.
2. **CH4MOD Parameters:** Upload daily $T_{\text{soil}}$, $T_{\text{air}}$, and water table files; configure $W_{\text{max}}$, Variety Index (VI), baseline soil organic matter (DSOM), and sand fraction.
3. **Water Management:** Design stepped flooding/drainage schedules (Period 1, 2, and 3).
4. **Mid-Season Additions:** Schedule top-dressing nitrogen fertilizer or organic matter token injections on designated days.
5. **Microbial Populations:** Adjust initial seed token allocations for each of the 10 taxa and chemical pools.
6. **Results & Visualization:**
   * **CH4MOD Model Tab:** Continuous crop biomass, water status, and total $\text{CH}_4$ emissions ($\text{mg/m}^2/\text{h}$).
   * **Microbiome Model Tab:** 10 individual taxon population curves, chemical pool turnover ($\text{SO}_4^{2-}$, acetate, $\text{NO}_3^-$), $\text{N}_2\text{O}$ and $\text{CO}_2$ gas fluxes, and final Petri Net token distribution maps.
   * **Export:** Hourly event log download in `.csv` format.

---

## 🛠️ Troubleshooting & Deployment Notes

### Hugging Face Spaces (ZeroGPU)
* When deploying to Hugging Face Spaces on a **ZeroGPU** tier, the simulation function is decorated with `@spaces.GPU` for runtime validation.
* **Recommended Tier:** Because this model runs on NumPy and PM4Py (pure CPU computation), it can be deployed on the **free CPU basic (2 vCPU, 16 GB RAM)** tier without requiring GPU hardware.

### Gradio 5+ Container Startup
* In containerized environments (Docker/Spaces), Gradio’s Server-Side Rendering (SSR) Node proxy may terminate unexpectedly. This repository explicitly sets `os.environ["GRADIO_SSR_MODE"] = "false"` and launches with `ssr_mode=False` to bind directly to the internal Uvicorn server on port `7860`.
* A patched `psutil.Process` handler is included to prevent container crashes when root PID is `0`.

---

## 📜 Citation

If you use this model or code in your research, please cite:

```bibtex
@software{CH4MicrobiomeSim,
  author       = {Kenny J.X. Lau, Kins Varghese, Ali Ma, Kasthurirengan Sampath, Goh Phuay Yee, Suhartono F. Wijoyo, Srinivasan Ramachandran, Naweed I. Naqvi1},
  title        = {Coupled CH4MOD and Metagenomically Constrained Petri Net Simulator for Rice Paddy Greenhouse Gas Emissions},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/kjxlau/CH4MicrobiomeSem}}
}
```

---
