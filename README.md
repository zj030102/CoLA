# CoLA


# Project Structure & Setup
> This repository is built based on OpenCOOD for our proposed **CoLA** method.

## File Locations
- Configuration file for CoLA: `./OpenCOOD/opencood/hypes_yaml/point_pillar_multi_baseline_deltafusion_ablation.yaml`
- Model entry script: `./OpenCOOD/opencood/models/point_pillar_deltafusion_ablation.py`
- Core operator directory: `./OpenCOOD/opencood/models/fuse_modules/delta`

## Environment
The basic environment setup follows the official OpenCOOD repository:
> https://github.com/DerrickXuNu/OpenCOOD

### Additional Dependencies
1. **RWKV7 Kernel**
The file `./opencood/models/fuse_modules/delta/rwkv7_fp32_hs32.cpp` requires pre-installed RWKV7 kernel.
It is loaded and invoked inside `opencood/models/fuse_modules/delta/pureDelta.py`.

2. **Mamba for Ablation Studies**
To run ablation experiments comparing against Mamba, install the provided wheel package:
```bash
pip install mamba_ssm-2.1.0+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
