#!/usr/bin/env python3

# Written 2019 by Ramon Witschi, ETH Computer Science BSc, for Bachelor Thesis @ CGL
# The function convert_am_to_raw was ported from https://www.csc.kth.se/~weinkauf/notes/amiramesh.html

# Nomenclature
#   BOLD usually means input arguments which are always expected, given the context.
#   grad is the gradient
#   a = jv, where j is the jacobian of v
#   b = grad(a)v

# Note
#   Irregardless of what was specified in the data .json, all filters for both a and b will be computed.
#   In particular, v, w_a, w_b and j_eigenvalues are always written out. The latter, since
#   computing the eigenvalues of j takes the longest and therefore benefits from being saved to disk.
#   The filters and IndeX scene .prj files will only be written for the specified w and filter types.

import os
import sys
import time
import json
import math
import pprint
import numpy as np
import pandas as pd
#import helper_functions as hf
import helper_functions as hf

max_float32 = np.finfo(dtype=np.float32).max - 1.0

PD = {

    # generally invariant parameters

    "DH": 1, # must be integer!
    "dimVec": 3,
    "dimC": 4,
    "TIME_STEP_LENGTH": 4,
    "VOLUME_FILE_EXTENSION": "raw",

    # Pre-initialized arguments that
    # don't have any effect on the filter,
    # in case of forgotten / unnecessary
    # (since maybe too specific) definitions
    # in the corresponding data .json files
    # This is more of a reminder what exists.

    "IMPORT": "am",
    "RECOMPUTE": True,
    "NORMALIZED": False,
    "TENSORPRODUCTS": True,

    "W_TYPE": "a, b",
    "LINE_TYPE": "no_line_filter",

    #LINE_TYPE s: "no_line_filter", "vortex_corelines"
    # Note: Leftovers from bifurcation lines and
    # vorticity extremal lines remain in the .py files;
    # however, they cannot really be pre-computed;
    # needs to be done in-kernel

    "S_RADIUS": "complex_eigenvalue",
    "S_COLOR": "complex_eigenvalue",
    "S_ALPHA": "complex_eigenvalue",

    "VXW_MAGNITUDE_THRESH_HIGH": max_float32,
    "SWIRLING_STRENGTH_THRESH_LOW": -1.0,
    "VELOCITY_MAGNITUDE_THRESH_LOW": -1.0,
    "VORTICITY_MAGNITUDE_THRESH_LOW": -1.0,

    "SWIRLING_JET_CENTER_DISTANCE_A_THRESH_HIGH": max_float32,
    "SWIRLING_JET_CENTER_DISTANCE_B_THRESH_HIGH": max_float32
}

'''------------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------------ get data specification ------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------'''

# left empty it means it's relative to the directory structure

json_path = ""

# please choose your desired volume file here

#json_file = "data_3D_Steady_Tornado.json"
#json_file = "data_3D_Steady_Stuart_Vortex.json"
#json_file = "data_2D_Unsteady_Moving_Center.json"
#json_file = "data_2D_Unsteady_Cylinder_Gauss3_Subset.json"
#json_file = "data_3D_Steady_Delta65_High.json"
#json_file = "data_3D_Unsteady_Swirling_Jet.json"
json_file = "data_3D_Unsteady_Borromean_Rings.json"

# load json file and update PD

json_file_path = json_path + json_file
json_dict = json.load(open(json_file_path, "rb"))
PD.update(json_dict)

'''------------------------------------------------------------------------------------------------------------------------'''
'''--------------------------------------------------- time series loop ---------------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------'''

print("\nPre-processing dataset with the following properties:\n")
pprint.pprint(PD)
print("")

for PD["time_step"] in range(PD["NB_TIME_STEPS"]):

    PD["print_count"] = 0
    PD["time_start"] = time.time()

    volume_name = PD["VOLUME_NAME"] + "_"
    PD["VOLUME_NAME"] = volume_name.replace("__", "_")

    # calculate time step number for the file input and output
    PD["current_iteration"] = PD["TIME_STEP_OFFSET"] + PD["time_step"] * PD["TIME_STEP_STRIDE"]
    PD["current_file_step_enum"] = str(PD["current_iteration"]).zfill(PD["TIME_STEP_LENGTH"])

    '''------------------------------------------------------------------------------------------------------------------------'''
    '''--------------------------------- import / generate v and if analytical: a, vxa, b, vxb --------------------------------'''
    '''------------------------------------------------------------------------------------------------------------------------'''

    if PD["RECOMPUTE"]:

        if "am" in PD["IMPORT"]: 
            PD = hf.convert_am_to_raw(PD)

        if "vti" in PD["IMPORT"]: 
            PD = hf.convert_vti_to_raw(PD)

        PD["dx"] = (abs(PD["maxX"] - PD["minX"])) / (PD["dimX"] - 1)
        PD["dy"] = (abs(PD["maxY"] - PD["minY"])) / (PD["dimY"] - 1)
        PD["dz"] = (abs(PD["maxZ"] - PD["minZ"])) / (PD["dimZ"] - 1)

        if "analytical" in PD["IMPORT"]:
            PD = hf.get_analytical_volumes(PD)
            PD["v_numerical"] = np.copy(PD["v_analytical"])

    else:

        dims_file_name = PD["VOLUME_NAME"] + "attributes"
        attr_json_dict = json.load(open(PD["VOLUME_DIR_PATH"] + "/" + dims_file_name + ".json", "rb"))
        PD.update(attr_json_dict)

        v_numerical = np.fromfile(hf.get_file_path(PD, "v"), dtype=np.float32)
        v_numerical = v_numerical.astype(np.float64)
        v_numerical = np.reshape(v_numerical, newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
        PD["v_numerical"] = v_numerical[:,:,:,0:3]

    PD["print_count"] = hf.write_benchmark(PD, "imported v / generated analytical v and volumes derived from it")

    '''------------------------------------------------------------------------------------------------------------------------'''
    '''-------------------- compute intermediate volumes: a, a_pv_line_filter, vxa, b, vxb, b_pv_line_filter ------------------'''
    '''------------------------------------------------------------------------------------------------------------------------'''

    PD = hf.compute_and_import_intermediate_volumes(PD)

    if "analytical" in PD["IMPORT"] and PD["RECOMPUTE"]:
        hf.verify_intermediate_volumes(PD)

    PD["print_count"] = hf.write_benchmark(PD, "computed intermediate volumes: a, a_pv_line_filter, vxa, b, vxb, b_pv_line_filter")

    '''------------------------------------------------------------------------------------------------------------------------'''
    '''---------------------------------------- calculate entries of all requested filters ------------------------------------'''
    '''------------------------------------------------------------------------------------------------------------------------'''

    PD = hf.get_s_volume(PD)
    PD["print_count"] = hf.write_benchmark(PD, "calculated entries of all requested filters")

    '''------------------------------------------------------------------------------------------------------------------------'''
    '''--------------------------------------------- export volumes, filters, etc. ---------------------------------------------'''
    '''------------------------------------------------------------------------------------------------------------------------'''

    hf.write_volumes(PD)
    if PD["time_step"] == 0:
        hf.write_index_scene_project(PD)
        hf.write_dimensions_regions_infinitesimals_json(PD)
    PD["print_count"] = hf.write_benchmark(PD, "exported volumes, filters, etc.")

    '''------------------------------------------------------------------------------------------------------------------------'''
    '''----------------------------------------------------- end of script ----------------------------------------------------'''
    '''------------------------------------------------------------------------------------------------------------------------'''
