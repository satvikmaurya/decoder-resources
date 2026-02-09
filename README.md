# `decoder-resources`

A framework for determining decoder resource requirements and scheduling decoding tasks using compiled IR.

This repository is complemented by additional bash scripts and datasets to reproduce the results of the paper - ["A Case for Elastic Quantum Error Correction Decoders"](https://arxiv.org/pdf/2406.17995), found in [Zenodo](https://doi.org/10.5281/zenodo.18555904).

## Using this repository

Clone using the following command:

`git clone --recursive https://github.com/satvikmaurya/decoder-resources.git`

## Docker image

### To build the image

`$ docker build -t $USER/vader .`

### To run the docker container

Make sure to create an additional directory before running the docker image: `mkdir -p $PWD/../LLI`
`$ docker-compose run app`

### To start stopped container

`$ docker start -ai [CONTAINER_ID]`

## Build `liblsqecc`

All environment variables required are set during the image build. The bash script `lsqecc_build.sh` can be run directly to build all required libraries for liblsqecc (a test fails at the end of the build, this can be ignored).

`$ bash lsqecc_build.sh`

## Generate program traces using `liblsqecc`

[Lattice Surgery Compiler](https://github.com/latticesurgery-com/liblsqecc) can be used for generating program traces used for determining decoder requirements. The following command is an example of how it can be used. Refer to the `liblsqecc` documentation for other configuration options.

`build/lsqecc_slicer -q -i input.qasm -t 1200 --graceful --printlli sliced -P wave --cnotcorrections always -L edpc --numlanes 1 --condensed --disttime 2 -o program_IR.lli`

## Using `decoder-resources`

`scripts/sim.py` is the primary simulation file. To see all available options, run:

`python3 sim.py --help`

## Citation

If you use the ideas or code of this work, please cite:

```bibtex
@misc{Maurya2024,
  doi = {10.48550/ARXIV.2406.17995},
  url = {https://arxiv.org/abs/2406.17995},
  author = {Maurya,  Satvik and Molavi,  Abtin and Albarghouthi,  Aws and Tannu,  Swamit},
  keywords = {Quantum Physics (quant-ph),  Hardware Architecture (cs.AR),  FOS: Physical sciences,  FOS: Physical sciences,  FOS: Computer and information sciences,  FOS: Computer and information sciences},
  title = {Managing Classical Processing Requirements for Quantum Error Correction},
  publisher = {arXiv},
  year = {2024},
  copyright = {Creative Commons Attribution Non Commercial No Derivatives 4.0 International}
}
```
