# Near Field Localization via AI-Aided Subspace Methods

## Related Publications
- [1] [DCD-MUSIC: Deep-Learning-Aided Cascaded Differentiable MUSIC Algorithm for Near-Field Localization of Multiple Sources](https://ieeexplore.ieee.org/abstract/document/10888295) (ICASSP 2025)
- [2] [Near Field Localization via AI-Aided Subspace Methods (preprint)](http://arxiv.org/abs/2504.00599)

## Introduction
This repository contain the implementation of the AI-aided subspace methods for near-field localizaiton:
- DCD-MUSIC
- NF-SSN

## Getting Started
### Prerequisites
- Python 3.10+
### Installation
- Init a new virtual environment
  - For conda, `conda env create -f full_environment.yml` and active it by `conda activate ai_subspace_env`
  - For venv, `py -m venv ai_subspace_env`
    - Activate the virtual environment
    - Go to [Pytorch official website](https://pytorch.org/) to install the correct version of Pytorch
    - Install the required packages by running `pip install -r requirements.txt`
- See example usage in 'main.py'

## Citation
If you find this work useful, please cite:

```bibtex
@inproceedings{gast2025dcdmusic,
  author    = {Arad Gast, Luc Le Magoarou, Nir Shlezinger},
  title     = {DCD-MUSIC: Deep-Learning-Aided Cascaded Differentiable MUSIC Algorithm for Near-Field Localization of Multiple Sources},
  booktitle = {ICASSP},
  year      = {2025},
  publisher = {IEEE}
}

@article{gast2025aisubspacenear,
  author    = {Arad Gast, Luc Le Magoarou, Nir Shlezinger},
  title     = {Near Field Localization via AI-Aided Subspace Methods},
  journal   = {arXiv preprint arXiv:2504.00599},
  year      = {2025}
}