This is a README for environment setup and inference, not for training so far.

### Installation Setup
```bash
conda create -n wow-dit-7b python=3.10 -y
pip install transformer-engine[pytorch]==1.12.0
git clone https://github.com/NVIDIA/apex
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex
pip install -r dit_models/wow-dit-7b/requires.txt
```

### Inference

```bash
cd scripts
conda activate wow-dit-7b
python infer_wow_dit_7b.py
```
### Installation Issues

* **Conda environment conflicts**: Create fresh environment with `conda create -n wow-dit-7b-clean python=3.10 -y`
* **Flash-attention build failures**: Install build tools with `apt-get install build-essential`
* **Transformer engine linking errors**: Reinstall with `pip install --force-reinstall transformer-engine==1.12.0`