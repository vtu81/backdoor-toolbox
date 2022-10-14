# SPECTRE

Codes are referenced from https://github.com/SewoongLab/spectre-defense, SPECTRE's official implementation (in Julia).

## Quick Start

All you need to do is to **make sure you have Julia installed and paths properly set**. Then install dependencies with:
```bash
# Run under this directory
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

To launch SPECTRE defense, simply run our drive code:
```bash
# Run under repository root
python cleanser.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.01 -cleanser=SPECTRE
```
