# Multi-Task Hypernetworks using PyTorch
This is the PyTorch source code for "Remote Sensing for Water Quality: A Multi-Task, Metadata-Driven Hypernetwork Approach" [IJCAI 2024]

### Included
We include the following algorithm implementations:

* Multi-Task Hypernetwork code - src\MTH\MultiTaskHypernetwork.py

* Hard sharing, Multilinear Relational Networks - src\MTL\MTL.py

* Cross-stitch Networks, Sluice Networks - src\MTL\CS.py

* Deep Multi-Task Representation Learning - src\MTL\TF.py

* Maximum Roaming Multi-Task Learning - src\MTL\MR.py

We also include a working Jupyter notebook example - example_experiments.ipynb

And the data generator for the synthetic datasets - generate_synthetic_datasets.ipynb

### Dependencies
Code was tested with:
- Python 3.8.5
- pytorch 1.7.1 with Cuda 11.0
- numpy 1.19.2

### Datasets
All datasets in the paper are available in the data folder.
