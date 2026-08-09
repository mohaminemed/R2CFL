# R2CFL: Robust Reputation-Driven Crowdsourced Federated Learning

## Overview

**Crowdsourced Federated Learning (CrowdFL)** extends traditional FL by enabling open and heterogeneous participation through a crowdsourcing paradigm. In such environments, reputation-based incentive mechanisms are commonly used to guide worker selection and improve trustworthiness.

While existing reputation-driven CrowdFL frameworks improve participant reliability, they largely overlook the robustness of reputation systems against stealthy adversaries capable of evading standard detection mechanisms. As a result, malicious participants may gradually accumulate reputation and gain increasing influence over future training tasks.

To address this challenge, we propose **R2CFL**, a **Robust Reputation-Driven Crowdsourced Federated Learning** framework. R2CFL introduces:

* A **robust reputation model** that continuously evaluates participant reliability.
* **R2-NNM (Robust Reputation-Aware Nearest Neighbor Mixing)**, a defense mechanism that couples reputation evolution with update filtering during aggregation.


Experimental results demonstrate that R2-NNM matches or outperforms state-of-the-art Byzantine and backdoor defenses under adaptive attackers. Furthermore, when combined with existing detect-and-filter defenses, the proposed reputation model accurately reflects their statistical performance by preserving true positive and false positive rates.

---

## Repository Contents

This repository contains the complete implementation of **R2-CFL** together with several state-of-the-art defense and reputation mechanisms used as baselines.

### Defense Mechanisms


* **M-Krum** (Blanchard et al., NeurIPS'17)
* **FLAME** (Nguyen et al., USENIX Security'22)
* **DeepSight** (Rieger et al., NDSS'22)
* **NNM** (Allouah et al., AISTATS'23)
* **R2-NNM** (proposed)

### Reputation Frameworks

* **AutoDFL** (Dif et al., NOMS'25)
* **SSMTD** (Peiming et al., Discover Computing'26)
* **R2CFL** (proposed)

---

## Supported Attacks

The framework supports the reproduction of experiments under three advanced adaptive attacks:

* **OMP** (Shejwalkar et al., NDSS'21)
* **A3FL** (Zhang et al., NeurIPS'23)
* **Neurotoxin** (Zhang et al., ICML'22)

**Note:** These attacks are implemented following their original specifications and can be combined with the available defense mechanisms and reputation models. The backdoor attack implementations used in this work are based on our [backdoor framework](https://github.com/Ayoub-46/sok_eswa).



---

## Installation

Clone the repository:

```bash
cd R2CFL
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

Experiments are executed using the provided shell script:

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

The script automatically iterates over combinations of:

* Defense mechanisms
* Datasets
* Attack scenarios

The corresponding configuration file is loaded from:

```text
src/experiment/configs/<defense>/<attack>_analysis_<dataset>.yml
```

### Example Configuration

The following configuration launches experiments using:

* Defense: `r2_nnm`
* Dataset: `gtsrb`
* Attack: `a3fl`

```bash
python main.py --config ./src/experiment/configs/r2_nnm/a3fl_analysis_gtsrb.yml
```

### Available Defenses

```text
r2_nnm      # Proposed R2CFL defense
nnm
flame
deepsight
autodfl
ssmtd
```

### Available Datasets

```text
gtsrb
femnist
cifar
fashionmnist
```

### Available Attacks

```text
omp
a3fl
neurotoxin
```

### Modifying Experiments

To evaluate additional combinations, edit the arrays in `run_experiment.sh`:

```bash
defenses=("r2_nnm" "flame" "deepsight")
datasets=("gtsrb" "cifar")
attacks=("omp" "a3fl" "neurotoxin")
```

The script will automatically execute all possible combinations and launch the corresponding configuration files.


Configuration files located in the `configs/` directory allow users to customize:

* Dataset
* Number of participants
* Attack type and parameters
* Defense mechanism and parameters
* Training hyperparameters



---

## Reproducing Paper Results

The repository provides scripts and configuration files to reproduce all experimental results presented in the paper.

The experiments include:

* Clean training performance (ACC/LOSS) analysis.
* Byzantine/Backdoor robustness (ACC/LOSS/ASR) analysis.
* Defense detection performance (TPR/FPR) analysis.
* Reputation convergence analysis.


---

## Paper

This work has been accepted for presentation at the **HotDiSec Workshop**, co-located with **ESORICS 2026**, to be held on **14--18 September 2026 in Rome, Italy**.

