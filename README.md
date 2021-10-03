Written 2019 by Ramon Witschi, ETH Computer Science BSc, for Bachelor Thesis @ CGL


Tested on Windows 10.

The environment variables should all be relative to the file structure.


We'll shortly explain the file structure:


./tensor_product_surface_visualization/*

Matlab code to visualize tensor_products. tensor_product_visualization.m is the main file, have fun.


./colormap_vorticity_magnitude.cmap

Used as a startup colormap. Specified in ./project-configs/scene_setup.prj


./cuda_kernel_xac_header.cuh

All helper functions, if changes are done in this file, IndeX has to be restarted.


./cuda_kernel_xac_parallel_vectors_operator.cuh

Main file, can be changed and updated on-the-fly in the IndeX Viewer (copy&paste code).


./data_*.json

Necessary for ./pipeline.py, specifies everything about the dataset.


./helper_functions.py

Self-explanatory


./pipeline.py

Preprocesses the datasets as specified in the corresponding ./data_*.json file.
