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

import json
import math
import os
import pprint
import re
import sys
import time

import numba
import numpy as np
import pandas as pd
import pyvista
import scipy
from scipy import ndimage

'''------------------------------------------------------------------------------------------------------------------------'''
'''----------------------------------------------- smaller helper functions -----------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------'''


@numba.njit
def evaluate_analytical_volume_3D(dimX, dimY, dimZ, dimVec, dx, dy, dz, minX, minY, minZ, f0, f1, f2):

    result = np.zeros((dimX, dimY, dimZ, dimVec), dtype=np.float64)

    for xx in range(dimX):
        for yy in range(dimY):
            for zz in range(dimZ):

                # affine pullback
                x = xx * dx + minX
                y = yy * dy + minY
                z = zz * dz + minZ

                result[xx, yy, zz, 0] = f0(x, y, z)
                result[xx, yy, zz, 1] = f1(x, y, z)
                result[xx, yy, zz, 2] = f2(x, y, z)

    return result


@numba.njit
def get_swirling_jet_center_distances(dimX, dimY, dimZ):

    swirling_jet_center_distances = np.zeros((dimX, dimY, dimZ))

    for x in range(dimX):
        for y in range(dimY):
            for z in range(dimZ):
                swirling_jet_center_distances[x, y, z] = math.sqrt((100.5 - x)**2 + (100.5 - z)**2)

    return swirling_jet_center_distances


@numba.njit
def get_j_numerical_eigenvalues(j_numerical, dimX, dimY, dimZ, nb_eigvals):

    j_numerical_eigenvalues = np.zeros((dimX, dimY, dimZ, nb_eigvals), dtype=np.complex128)

    for x in range(dimX):
        for y in range(dimY):
            for z in range(dimZ):
                j_numerical_eigenvalues[x, y, z] = np.linalg.eigvals(j_numerical[x, y, z])

    return j_numerical_eigenvalues


def get_scalar_gradient(v, dimX, dimY, dimZ, dimVec, dx, dy, dz):

    g0 = np.gradient(v, dx, dy, dz, edge_order=2, axis=(0, 1, 2))
    g = np.zeros((dimX, dimY, dimZ, dimVec), dtype=np.float64)

    g[:, :, :, 0] = g0[0]
    g[:, :, :, 1] = g0[1]
    g[:, :, :, 2] = g0[2]

    return g


def get_jacobian_finite_differences(v, dimX, dimY, dimZ, dimVec, dx, dy, dz):

    fxyz = get_scalar_gradient(v[:, :, :, 0], dimX, dimY, dimZ, dimVec, dx, dy, dz)
    gxyz = get_scalar_gradient(v[:, :, :, 1], dimX, dimY, dimZ, dimVec, dx, dy, dz)
    hxyz = get_scalar_gradient(v[:, :, :, 2], dimX, dimY, dimZ, dimVec, dx, dy, dz)
    j = np.zeros((dimX, dimY, dimZ, dimVec, dimVec), dtype=np.float64)

    j[:, :, :, 0, 0] = fxyz[:, :, :, 0]
    j[:, :, :, 0, 1] = fxyz[:, :, :, 1]
    j[:, :, :, 0, 2] = fxyz[:, :, :, 2]
    j[:, :, :, 1, 0] = gxyz[:, :, :, 0]
    j[:, :, :, 1, 1] = gxyz[:, :, :, 1]
    j[:, :, :, 1, 2] = gxyz[:, :, :, 2]
    j[:, :, :, 2, 0] = hxyz[:, :, :, 0]
    j[:, :, :, 2, 1] = hxyz[:, :, :, 1]
    j[:, :, :, 2, 2] = hxyz[:, :, :, 2]

    return j


def get_matvec(A, b, dimX, dimY, dimZ, dimVec):

    c = np.zeros((dimX, dimY, dimZ, dimVec), dtype=np.float64)

    c[:, :, :, 0] = b[:, :, :, 0]*A[:, :, :, 0, 0] + b[:, :, :, 1]*A[:, :, :, 0, 1] + b[:, :, :, 2]*A[:, :, :, 0, 2]
    c[:, :, :, 1] = b[:, :, :, 0]*A[:, :, :, 1, 0] + b[:, :, :, 1]*A[:, :, :, 1, 1] + b[:, :, :, 2]*A[:, :, :, 1, 2]
    c[:, :, :, 2] = b[:, :, :, 0]*A[:, :, :, 2, 0] + b[:, :, :, 1]*A[:, :, :, 2, 1] + b[:, :, :, 2]*A[:, :, :, 2, 2]

    return c


def compare_volumes(PD, vol1, vol2, name):

    absdiff = np.abs(np.subtract(vol1.astype(dtype=np.float32), vol2.astype(dtype=np.float32)))
    vol3 = pd.DataFrame()
    vol3.insert(loc=0, column="u", value=absdiff[:, :, :, 0].flatten(), allow_duplicates=True)
    vol3.insert(loc=1, column="v", value=absdiff[:, :, :, 1].flatten(), allow_duplicates=True)
    vol3.insert(loc=2, column="w", value=absdiff[:, :, :, 2].flatten(), allow_duplicates=True)
    vol3_stats = vol3.describe(include='all')
    vol3_stats.to_csv(PD["VOLUME_DIR_PATH"] + "/comparison_" + name + ".csv")
    print("")
    print(name)
    print("")
    print(vol3_stats.to_string())
    print("")


def analyze_single_vector_across_volume(PD, vol1, column_name, name):

    vol = vol1.astype(dtype=np.float32)
    vol3 = pd.DataFrame()
    vol3.insert(loc=0, column=column_name, value=vol.flatten(), allow_duplicates=False)
    vol3_stats = vol3.describe(include='all')
    vol3_stats.to_csv(PD["VOLUME_DIR_PATH"] + "/" + PD["VOLUME_NAME"] + "_analysed_" + name + ".csv")
    print("")
    print(name)
    print("")
    print(vol3_stats.to_string())
    print("")


def get_file_path(PD, volume_name):

    file_name_path_template = PD["VOLUME_DIR_PATH"] + "/" + PD["VOLUME_NAME"] + "<#file_name>_" + PD["current_file_step_enum"] + "." + PD["VOLUME_FILE_EXTENSION"]
    return file_name_path_template.replace("<#file_name>", volume_name)


'''------------------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------- volume data generation / import -------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------'''


def get_analytical_volumes(PD):

    dimX = PD["dimX"]
    dimY = PD["dimY"]
    dimZ = PD["dimZ"]
    dimVec = PD["dimVec"]

    dx = PD["dx"]
    dy = PD["dy"]
    dz = PD["dz"]

    minX = PD["minX"]
    minY = PD["minY"]
    minZ = PD["minZ"]

    lambda_str = "lambda x,y,z: "
    fV0 = numba.jit(nopython=True)(eval(lambda_str + PD["V0"]))
    fV1 = numba.jit(nopython=True)(eval(lambda_str + PD["V1"]))
    fV2 = numba.jit(nopython=True)(eval(lambda_str + PD["V2"]))

    fA0 = numba.jit(nopython=True)(eval(lambda_str + PD["A0"]))
    fA1 = numba.jit(nopython=True)(eval(lambda_str + PD["A1"]))
    fA2 = numba.jit(nopython=True)(eval(lambda_str + PD["A2"]))

    fVXA0 = numba.jit(nopython=True)(eval(lambda_str + PD["VXA0"]))
    fVXA1 = numba.jit(nopython=True)(eval(lambda_str + PD["VXA1"]))
    fVXA2 = numba.jit(nopython=True)(eval(lambda_str + PD["VXA2"]))

    fB0 = numba.jit(nopython=True)(eval(lambda_str + PD["B0"]))
    fB1 = numba.jit(nopython=True)(eval(lambda_str + PD["B1"]))
    fB2 = numba.jit(nopython=True)(eval(lambda_str + PD["B2"]))

    fVXB0 = numba.jit(nopython=True)(eval(lambda_str + PD["VXB0"]))
    fVXB1 = numba.jit(nopython=True)(eval(lambda_str + PD["VXB1"]))
    fVXB2 = numba.jit(nopython=True)(eval(lambda_str + PD["VXB2"]))

    PD["v_analytical"] = evaluate_analytical_volume_3D(dimX, dimY, dimZ, dimVec, dx, dy, dz, minX, minY, minZ, fV0, fV1, fV2)
    PD["a_analytical"] = evaluate_analytical_volume_3D(dimX, dimY, dimZ, dimVec, dx, dy, dz, minX, minY, minZ, fA0, fA1, fA2)
    PD["vxa_analytical"] = evaluate_analytical_volume_3D(dimX, dimY, dimZ, dimVec, dx, dy, dz, minX, minY, minZ, fVXA0, fVXA1, fVXA2)
    PD["b_analytical"] = evaluate_analytical_volume_3D(dimX, dimY, dimZ, dimVec, dx, dy, dz, minX, minY, minZ, fB0, fB1, fB2)
    PD["vxb_analytical"] = evaluate_analytical_volume_3D(dimX, dimY, dimZ, dimVec, dx, dy, dz, minX, minY, minZ, fVXB0, fVXB1, fVXB2)

    return PD


def convert_am_to_raw(PD):

    file_path = PD["VOLUME_DIR_PATH"] + "/" + PD["VOLUME_NAME"] + PD["current_file_step_enum"] + ".am"

    with open(file_path, "rb") as fp:
        binary_buf = fp.read(2047)
        buffer = str(binary_buf)
        print("\n", buffer, "\n")
        if not buffer.find("# AmiraMesh BINARY-LITTLE-ENDIAN 2.1"):
            print("Not a proper AmiraMesh file.\n")
            return 1

        # Is it a uniform grid?
        # We need this only for the sanity check below.
        PD["bIsUniform"] = (
            buffer.find("CoordType \"uniform\"") != 0)

        PD["grid_type"] = "UNKNOWN"
        if PD["bIsUniform"]:
            PD["grid_type"] = "uniform"

        # Here we parse both the grid dimension and bounding box values
        buf_start = buffer.find("define Lattice")
        numeric_values = re.findall(
            r"[+]?\d+", buffer[buf_start:])

        PD["dimX"] = int(numeric_values[0])
        PD["dimY"] = int(numeric_values[1])
        PD["dimZ"] = int(numeric_values[2])

        buf_start = buffer.find("Lattice { float")
        numeric_values = re.findall(
            r"[+]?\d+", buffer[buf_start:])

        PD["dimVec_import"] = int(numeric_values[0])
        if (buffer.find("Lattice { float Data }") != -1):
            PD["dimVec_import"] = 1

        buf_start = buffer.find("BoundingBox ")
        numeric_values = re.findall(r"[+-]?\d+\.\d+|[+-]?\d+", buffer[buf_start:])

        PD["minX"] = float(numeric_values[0])
        PD["maxX"] = float(numeric_values[1])

        PD["minY"] = float(numeric_values[2])
        PD["maxY"] = float(numeric_values[3])

        PD["minZ"] = float(numeric_values[4])
        PD["maxZ"] = float(numeric_values[5])

        if "L2LES_4" in PD["VOLUME_NAME"]:

            PD["minX"] = 0.0
            PD["maxX"] = 2.0

            PD["minY"] = 0.0
            PD["maxY"] = 1.0

            PD["minZ"] = 0.0
            PD["maxZ"] = 1.0

        # Sanity check
        if PD["dimX"] <= 0\
            or PD["dimY"] <= 0\
            or PD["dimZ"] <= 0\
            or PD["minX"] > PD["maxX"]\
            or PD["minY"] > PD["maxY"]\
            or PD["minZ"] > PD["maxZ"]\
            or not PD["bIsUniform"]\
                or PD["dimVec_import"] <= 0:
            print("Something went wrong\n")
            exit(1)

        # Find the beginning of the data section
        data_parse = "# Data section follows"
        index_start_data = binary_buf.find(b"# Data section follows")

        # Set the file pointer to the beginning of the binary data section
        fp.seek(index_start_data, 0)

        # Consume this line, which is "# Data section follows"
        buffer1 = fp.readline(2047)
        print(buffer1)

        # Consume the next line, which is "@1"
        buffer2 = fp.readline(2047)
        print(buffer2)

        if not index_start_data > 0:
            print("error:'", data_parse, "' couldn't be found.\n")
            exit(1)

        # Read the data
        PD["num_total_components"] = PD["dimX"]*PD["dimY"]*PD["dimZ"]*PD["dimVec_import"]

        v_import = np.fromfile(fp, dtype=np.float32, count=PD["num_total_components"])

    idx = 0
    v = np.zeros(shape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimVec_import"]), dtype=np.float64)
    for z in range(PD["dimZ"]):
        for y in range(PD["dimY"]):
            for x in range(PD["dimX"]):

                index_x_fastest = ((z*PD["dimY"] + y)*PD["dimX"] + x)*PD["dimVec_import"]

                assert(v_import[index_x_fastest] == v_import[idx*PD["dimVec_import"]])

                v0 = v_import[index_x_fastest + 0]
                v1 = v_import[index_x_fastest + 1]
                v2 = 1.0

                if PD["dimVec_import"] == 3:
                    v2 = v_import[index_x_fastest + 2]

                if PD["NORMALIZED"]:
                    v_norm = math.sqrt(v0*v0 + v1*v1 + v2*v2)
                    v_norm = v_norm if (v_norm) else 1.0
                    v0 = v0 / v_norm
                    v1 = v1 / v_norm
                    v2 = v2 / v_norm

                v[x, y, z, 0] = v0
                v[x, y, z, 1] = v1
                v[x, y, z, 2] = v2

                idx += 1

    PD["v_numerical"] = v

    return PD


def convert_vti_to_raw(PD):

    data = pyvista.read(PD["VOLUME_DIR_PATH"] + "/" + PD["VOLUME_NAME"] + PD["current_file_step_enum"] + ".vti")

    PD["bIsUniform"] = True
    PD["dimVec_import"] = data.n_arrays

    PD["dimX"] = data.dimensions[0]
    PD["dimY"] = data.dimensions[1]
    PD["dimZ"] = data.dimensions[2]

    PD["minX"] = data.extent[0]
    PD["maxX"] = data.extent[1]

    PD["minY"] = data.extent[2]
    PD["maxY"] = data.extent[3]

    PD["minZ"] = data.extent[4]
    PD["maxZ"] = data.extent[5]

    # Sanity check
    if PD["dimX"] <= 0\
        or PD["dimY"] <= 0\
        or PD["dimZ"] <= 0\
        or PD["minX"] > PD["maxX"]\
        or PD["minY"] > PD["maxY"]\
        or PD["minZ"] > PD["maxZ"]\
        or not PD["bIsUniform"]\
            or PD["dimVec_import"] <= 1:

        print("Something went wrong\n")
        exit(1)

    point_arrays = data.point_arrays
    v_volume = np.zeros((PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimVec"]), dtype=np.double)

    for x in range(PD["dimX"]):
        for y in range(PD["dimY"]):
            for z in range(PD["dimZ"]):

                index_x_fastest = ((z*PD["dimY"] + y)*PD["dimX"] + x)

                v0 = point_arrays['u'][index_x_fastest]
                v1 = point_arrays['v'][index_x_fastest]
                v2 = 1.0

                if PD["dimVec_import"] == 3:
                    v2 = point_arrays['w'][index_x_fastest]

                if "flow_t" in PD["VOLUME_NAME"]:  # subtract a constant "ambient flow field" from the first component
                    v0 = v0 - 0.83

                if PD["NORMALIZED"]:
                    v_norm = math.sqrt(v0*v0 + v1*v1 + v2*v2)
                    v_norm = (v_norm) if (v_norm) else (1.0)
                    v0 = v0 / v_norm
                    v1 = v1 / v_norm
                    v2 = v2 / v_norm

                v_volume[x, y, z, 0] = v0
                v_volume[x, y, z, 1] = v1
                v_volume[x, y, z, 2] = v2

    PD["v_numerical"] = v_volume

    return PD


def compute_and_import_intermediate_volumes(PD):

    dimX = PD["dimX"]
    dimY = PD["dimY"]
    dimZ = PD["dimZ"]
    dimVec = PD["dimVec"]

    dx = PD["dx"]
    dy = PD["dy"]
    dz = PD["dz"]

    PD["j_numerical"] = get_jacobian_finite_differences(PD["v_numerical"], dimX, dimY, dimZ, dimVec, dx, dy, dz)

    if PD["RECOMPUTE"]:
        PD["j_numerical_eigenvalues"] = get_j_numerical_eigenvalues(PD["j_numerical"].astype(np.complex128), PD["dimX"], PD["dimY"], PD["dimZ"], 3)
        PD["print_count"] = write_benchmark(PD, "computed eigenvalues of j_numerical")

    else:
        PD["j_numerical_eigenvalues"] = np.fromfile(get_file_path(PD, "j_eigenvalues"), dtype=np.complex128)
        PD["j_numerical_eigenvalues"] = PD["j_numerical_eigenvalues"].astype(np.complex128)
        PD["j_numerical_eigenvalues"] = np.reshape(PD["j_numerical_eigenvalues"], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], 3))
        PD["print_count"] = write_benchmark(PD, "imported eigenvalues of j_numerical")

    PD["print_count"] = write_benchmark(PD, "computed j_numerical")

    if PD["RECOMPUTE"]:
        PD["a_numerical"] = get_matvec(PD["j_numerical"], PD["v_numerical"], dimX, dimY, dimZ, dimVec)
        PD["print_count"] = write_benchmark(PD, "computed a_numerical")

    else:
        a_numerical = np.fromfile(get_file_path(PD, "w_a"), dtype=np.float32)
        a_numerical = a_numerical.astype(np.float64)
        a_numerical = np.reshape(a_numerical, newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
        PD["a_numerical"] = a_numerical[:, :, :, 0:3]
        PD["print_count"] = write_benchmark(PD, "imported a_numerical")

    if PD["RECOMPUTE"]:
        PD["a_pv_line_tensorprod_filter"] = get_may_contain_pv_lines(PD["v_numerical"], PD["a_numerical"], PD["dimX"], PD["dimY"], PD["dimZ"], PD["DH"])
        PD["a_pv_line_tensorprod_filter"] = PD["a_pv_line_tensorprod_filter"].astype(np.bool)
        PD["a_pv_line_tensorprod_filter"].tofile(get_file_path(PD, "a_pv_line_tensorprod_filter"), sep="", format="%s")
        print("\na_pv_line_tensorprod_filter   # of 1.0 entries : {}/{}\n".format(np.sum(PD["a_pv_line_tensorprod_filter"]), PD["dimX"]*PD["dimY"]*PD["dimZ"]))
        PD["print_count"] = write_benchmark(PD, "computed a_pv_line_tensorprod_filter")

    else:
        PD["a_pv_line_tensorprod_filter"] = np.fromfile(get_file_path(PD, "a_pv_line_tensorprod_filter"), dtype=np.bool)
        PD["a_pv_line_tensorprod_filter"] = np.reshape(PD["a_pv_line_tensorprod_filter"], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"]))
        print("\na_pv_line_tensorprod_filter   # of 1.0 entries : {}/{}\n".format(np.sum(PD["a_pv_line_tensorprod_filter"]), PD["dimX"]*PD["dimY"]*PD["dimZ"]))
        PD["print_count"] = write_benchmark(PD, "imported a_pv_line_tensorprod_filter")

    PD["vxa_numerical"] = np.cross(PD["v_numerical"], PD["a_numerical"])
    PD["print_count"] = write_benchmark(PD, "computed vxa")

    PD["grada_numerical"] = get_jacobian_finite_differences(PD["a_numerical"], dimX, dimY, dimZ, dimVec, dx, dy, dz)
    PD["print_count"] = write_benchmark(PD, "computed grada_numerical")

    if PD["RECOMPUTE"]:
        PD["b_numerical"] = get_matvec(PD["grada_numerical"], PD["v_numerical"], dimX, dimY, dimZ, dimVec)
        PD["print_count"] = write_benchmark(PD, "computed b_numerical")

    else:
        b_numerical = np.fromfile(get_file_path(PD, "w_b"), dtype=np.float32)
        b_numerical = b_numerical.astype(np.float64)
        b_numerical = np.reshape(b_numerical, newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
        PD["b_numerical"] = b_numerical[:, :, :, 0:3]
        PD["print_count"] = write_benchmark(PD, "imported b_numerical")

    PD["vxb_numerical"] = np.cross(PD["v_numerical"], PD["b_numerical"])
    PD["print_count"] = write_benchmark(PD, "computed vxb")

    if PD["RECOMPUTE"]:
        PD["b_pv_line_tensorprod_filter"] = get_may_contain_pv_lines(PD["v_numerical"], PD["b_numerical"], PD["dimX"], PD["dimY"], PD["dimZ"], PD["DH"])
        PD["b_pv_line_tensorprod_filter"] = PD["b_pv_line_tensorprod_filter"].astype(np.bool)
        PD["b_pv_line_tensorprod_filter"].tofile(get_file_path(PD, "b_pv_line_tensorprod_filter"), sep="", format="%s")
        print("\nb_pv_line_tensorprod_filter     # of 1.0 entries : {}/{}\n".format(np.sum(PD["b_pv_line_tensorprod_filter"]), PD["dimX"]*PD["dimY"]*PD["dimZ"]))
        PD["print_count"] = write_benchmark(PD, "computed b_pv_line_tensorprod_filter")

    else:
        PD["b_pv_line_tensorprod_filter"] = np.fromfile(get_file_path(PD, "b_pv_line_tensorprod_filter"), dtype=np.bool)
        PD["b_pv_line_tensorprod_filter"] = PD["b_pv_line_tensorprod_filter"].astype(np.bool)
        PD["b_pv_line_tensorprod_filter"] = np.reshape(PD["b_pv_line_tensorprod_filter"], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"]))
        print("\nb_pv_line_tensorprod_filter     # of 1.0 entries : {}/{}\n".format(np.sum(PD["b_pv_line_tensorprod_filter"]), PD["dimX"]*PD["dimY"]*PD["dimZ"]))
        PD["print_count"] = write_benchmark(PD, "imported b_pv_line_tensorprod_filter")

    if "swirling-jet" in PD["VOLUME_NAME"]:
        PD["swirling_jet_center_distances"] = get_swirling_jet_center_distances(PD["dimX"], PD["dimY"], PD["dimZ"])
        PD["print_count"] = write_benchmark(PD, "calculated swirling_jet_distances")

    return PD


def verify_intermediate_volumes(PD):

    compare_volumes(PD, PD["a_analytical"], PD["a_numerical"], "a analytical vs finite differences")
    compare_volumes(PD, PD["vxa_analytical"], PD["vxa_numerical"], "vxa analytical vs finite differences")
    compare_volumes(PD, PD["b_analytical"], PD["b_numerical"], "b analytical vs finite differences")
    compare_volumes(PD, PD["vxb_analytical"], PD["vxb_numerical"], "vxb analytical vs finite differences")


'''------------------------------------------------------------------------------------------------------------------------'''
'''------------------------------------------ import and export of volumes and co -----------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------'''


def write_volumes(PD):

    PD["j_numerical_eigenvalues"].tofile(get_file_path(PD, "j_eigenvalues"), sep="", format="%s")

    v_out = np.zeros((PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]), dtype=np.float64)
    v_out[:, :, :, 0:3] = PD["v_numerical"]
    v_out = v_out.astype(dtype=np.float32)
    v_out.tofile(get_file_path(PD, "v"), sep="", format="%s")

    w_a_out = np.zeros((PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]), dtype=np.float64)
    w_a_out[:, :, :, 0:3] = PD["a_numerical"]
    w_a_out = w_a_out.astype(dtype=np.float32)
    w_a_out.tofile(get_file_path(PD, "w_a"), sep="", format="%s")

    w_b_out = np.zeros((PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]), dtype=np.float64)
    w_b_out[:, :, :, 0:3] = PD["b_numerical"]
    w_b_out = w_b_out.astype(dtype=np.float32)
    w_b_out.tofile(get_file_path(PD, "w_b"), sep="", format="%s")

    if "a" in PD["W_TYPE"]:

        if "no_line_filter" in PD["LINE_TYPE"]:
            s_a_no_line_filter_out = PD["s_a_no_line_filter"].astype(np.float32)
            s_a_no_line_filter_out.tofile(get_file_path(PD, "s_a_no_line_filter"), sep="", format="%s")

        if "vortex_corelines" in PD["LINE_TYPE"]:
            s_a_vortex_corelines_out = PD["s_a_vortex_corelines"].astype(np.float32)
            s_a_vortex_corelines_out.tofile(get_file_path(PD, "s_a_vortex_corelines"), sep="", format="%s")

        if "bifurcation_lines" in PD["LINE_TYPE"]:
            s_a_bifurcation_lines_out = PD["s_a_bifurcation_lines"].astype(np.float32)
            s_a_bifurcation_lines_out.tofile(get_file_path(PD, "s_a_bifurcation_lines"), sep="", format="%s")

        if "vorticity_extremal_lines" in PD["LINE_TYPE"]:
            s_a_vorticity_extremal_lines_out = PD["s_a_vorticity_extremal_lines"].astype(np.float32)
            s_a_vorticity_extremal_lines_out.tofile(get_file_path(PD, "s_a_vorticity_extremal_lines"), sep="", format="%s")

    if "b" in PD["W_TYPE"]:

        if "no_line_filter" in PD["LINE_TYPE"]:
            s_b_no_line_filter_out = PD["s_b_no_line_filter"].astype(np.float32)
            s_b_no_line_filter_out.tofile(get_file_path(PD, "s_b_no_line_filter"), sep="", format="%s")

        if "vortex_corelines" in PD["LINE_TYPE"]:
            s_b_vortex_corelines_out = PD["s_b_vortex_corelines"].astype(np.float32)
            s_b_vortex_corelines_out.tofile(get_file_path(PD, "s_b_vortex_corelines"), sep="", format="%s")

        if "bifurcation_lines" in PD["LINE_TYPE"]:
            s_b_bifurcation_lines_out = PD["s_b_bifurcation_lines"].astype(np.float32)
            s_b_bifurcation_lines_out.tofile(get_file_path(PD, "s_b_bifurcation_lines"), sep="", format="%s")

        if "vorticity_extremal_lines" in PD["LINE_TYPE"]:
            s_b_vorticity_extremal_lines_out = PD["s_b_vorticity_extremal_lines"].astype(np.float32)
            s_b_vorticity_extremal_lines_out.tofile(get_file_path(PD, "s_b_vorticity_extremal_lines"), sep="", format="%s")


def write_benchmark(PD, name):

    file_name = PD["VOLUME_NAME"] + "benchmark"
    file_path = PD["VOLUME_DIR_PATH"] + "/" + file_name + ".txt"

    if PD["time_step"] == 0 and PD["print_count"] == 0:
        with open(file_path, "w") as fp:
            fp.write(PD["VOLUME_NAME"] + "\n\n")

    txt = "Stage\t {:2}\t Elapsed Time \t{:5.3}\t{}".format(PD["print_count"], time.time() - PD["time_start"], name)

    with open(file_path, "a") as fp:
        fp.write(txt)

    print(txt)

    return PD["print_count"] + 1


def print_filter_information(PD, count, filtered_volume, filter_name, filter_reason):

    file_name = PD["VOLUME_NAME"] + "filter_information"
    file_path = PD["VOLUME_DIR_PATH"] + "/" + file_name + ".txt"

    if PD["time_step"] == 0 and count == 0:
        with open(file_path, "w") as fp:
            fp.write(PD["VOLUME_NAME"] + "\n\n")

    this_vol = np.array(filtered_volume, copy=True)
    this_sum = np.sum(this_vol)
    txt = "\n{:5} npsum of {:35} after {:>35} : {:8}".format(count, filter_name, filter_reason, this_sum)

    with open(file_path, "a") as fp:
        fp.write(txt)

    print(txt)

    return count + 1


def write_dimensions_regions_infinitesimals_json(PD):

    file_name = PD["VOLUME_NAME"] + "attributes"

    with open(PD["VOLUME_DIR_PATH"] + "/" + file_name + ".json", "w") as fp:

        print("{", file=fp)
        print("\t\"NORMALIZED\": \"" + str(PD["NORMALIZED"]) + "\",", file=fp)
        print("\t\"TIME_STEP_LENGTH\": " + str(PD["TIME_STEP_LENGTH"]) + ",", file=fp)
        print("\t\"VOLUME_FILE_EXTENSION\": \"" + PD["VOLUME_FILE_EXTENSION"] + "\",", file=fp)
        print("\t\"VOLUME_DATA_TYPE\": \"" + str(np.float32) + "\",", file=fp)
        print("", file=fp)
        print("\t\"dimX\": " + str(PD["dimX"]) + ",", file=fp)
        print("\t\"dimY\": " + str(PD["dimY"]) + ",", file=fp)
        print("\t\"dimZ\": " + str(PD["dimZ"]) + ",", file=fp)
        print("\t\"dimC\": " + str(PD["dimC"]) + ",", file=fp)
        print("", file=fp)
        print("\t\"minX\": " + str(PD["minX"]) + ",", file=fp)
        print("\t\"minY\": " + str(PD["minY"]) + ",", file=fp)
        print("\t\"minZ\": " + str(PD["minZ"]) + ",", file=fp)
        print("", file=fp)
        print("\t\"maxX\": " + str(PD["maxX"]) + ",", file=fp)
        print("\t\"maxY\": " + str(PD["maxY"]) + ",", file=fp)
        print("\t\"maxZ\": " + str(PD["maxZ"]) + ",", file=fp)
        print("", file=fp)
        print("\t\"dx\": " + str(PD["dx"]) + ",", file=fp)
        print("\t\"dy\": " + str(PD["dy"]) + ",", file=fp)
        print("\t\"dz\": " + str(PD["dz"]), file=fp)
        print("}", file=fp)


def write_index_scene_project(PD):

    txt = index_prj_txt()
    txt = txt.replace("<#zyx_to_xyz>", "true")

    if "L2LES_4" in PD["VOLUME_NAME"]:
        txt = txt.replace("<#ROI>", str(PD["dimX"]) + " " + str(PD["dimY"]) + " " + str(PD["dimZ"] - 7))

    elif "flow_t" in PD["VOLUME_NAME"]:
        txt = txt.replace("<#ROI>", str(PD["dimX"]) + " " + str(PD["dimY"]) + " " + str(PD["dimZ"]))  # TODO

    else:
        txt = txt.replace("<#ROI>", str(PD["dimX"]) + " " + str(PD["dimY"]) + " " + str(PD["dimZ"]))

    txt = txt.replace("<#spatial_dimensions>", str(PD["dimX"]) + " " + str(PD["dimY"]) + " " + str(PD["dimZ"]))
    txt = txt.replace("<#spatial_dimensions_subcube>", str(PD["dimX"]) + " " + str(PD["dimY"]) + " " + str(PD["dimZ"]))
    txt = txt.replace("<#interval_seconds>", "0.0")
    txt = txt.replace("<#nb_time_steps>", str(PD["NB_TIME_STEPS"]))
    txt = txt.replace("<#time_step_enum_stride>", str(PD["TIME_STEP_STRIDE"]))
    txt = txt.replace("<#time_step_enum_offset>", str(PD["TIME_STEP_OFFSET"]))
    txt = txt.replace("<#volume_dir_path>", PD["VOLUME_DIR_PATH"])
    txt = txt.replace("<#volume_name_v>", PD["VOLUME_NAME"] + "v_")

    if "a" in PD["W_TYPE"]:

        txt_jv = txt.replace("<#volume_name_w>", PD["VOLUME_NAME"] + "w_a_")

        if "no_line_filter" in PD["LINE_TYPE"]:
            txt_out = txt_jv.replace("<#volume_name_s>", PD["VOLUME_NAME"] + "s_a_no_line_filter_")
            file_name_out = "scene_" + PD["VOLUME_NAME"] + "w_a_" + "no_line_filter"
            with open(file_name_out + ".prj", "w") as fp:
                fp.write(txt_out)

        if "vortex_corelines" in PD["LINE_TYPE"]:
            txt_out = txt_jv.replace("<#volume_name_s>", PD["VOLUME_NAME"] + "s_a_vortex_corelines_")
            file_name_out = "scene_" + PD["VOLUME_NAME"] + "w_a_" + "vortex_corelines"
            with open(file_name_out + ".prj", "w") as fp:
                fp.write(txt_out)

        if "bifurcation_lines" in PD["LINE_TYPE"]:
            txt_out = txt_jv.replace("<#volume_name_s>", PD["VOLUME_NAME"] + "s_a_bifurcation_lines_")
            file_name_out = "scene_" + PD["VOLUME_NAME"] + "w_a_" + "bifurcation_lines"
            with open(file_name_out + ".prj", "w") as fp:
                fp.write(txt_out)

        if "vorticity_extremal_lines" in PD["LINE_TYPE"]:
            txt_out = txt_jv.replace("<#volume_name_s>", PD["VOLUME_NAME"] + "s_a_vorticity_extremal_lines_")
            file_name_out = "scene_" + PD["VOLUME_NAME"] + "w_a_" + "vorticity_extremal_lines"
            with open(file_name_out + ".prj", "w") as fp:
                fp.write(txt_out)

    if "b" in PD["W_TYPE"]:

        txt_gjvv = txt.replace("<#volume_name_w>", PD["VOLUME_NAME"] + "w_b_")

        if "no_line_filter" in PD["LINE_TYPE"]:
            txt_out = txt_gjvv.replace("<#volume_name_s>", PD["VOLUME_NAME"] + "s_b_no_line_filter_")
            file_name_out = "scene_" + PD["VOLUME_NAME"] + "w_b_" + "no_line_filter"
            with open(file_name_out + ".prj", "w") as fp:
                fp.write(txt_out)

        if "vortex_corelines" in PD["LINE_TYPE"]:
            txt_out = txt_gjvv.replace("<#volume_name_s>", PD["VOLUME_NAME"] + "s_b_vortex_corelines_")
            file_name_out = "scene_" + PD["VOLUME_NAME"] + "w_b_" + "vortex_corelines"
            with open(file_name_out + ".prj", "w") as fp:
                fp.write(txt_out)

        if "bifurcation_lines" in PD["LINE_TYPE"]:
            txt_out = txt_gjvv.replace("<#volume_name_s>", PD["VOLUME_NAME"] + "s_b_bifurcation_lines_")
            file_name_out = "scene_" + PD["VOLUME_NAME"] + "w_b_" + "bifurcation_lines"
            with open(file_name_out + ".prj", "w") as fp:
                fp.write(txt_out)

        if "vorticity_extremal_lines" in PD["LINE_TYPE"]:
            txt_out = txt_gjvv.replace("<#volume_name_s>", PD["VOLUME_NAME"] + "s_b_vorticity_extremal_lines_")
            file_name_out = "scene_" + PD["VOLUME_NAME"] + "w_b_" + "vorticity_extremal_lines"
            with open(file_name_out + ".prj", "w") as fp:
                fp.write(txt_out)


def index_prj_txt():

    return "#! index_app_project 0\n"\
        + "# global setup\n"\
        + "index::region_of_interest                                     = 0 0 0 <#ROI>\n\n"\
        + "# subregion setup\n"\
        + "index::subcube_size                                           = <#spatial_dimensions_subcube>\n"\
        + "index::subcube_border_size                                    = 0\n\n"\
        + "#------------------------------------------------------------\n"\
        + "### Time Step\n\n"\
        + "# time series total clock interval (in Seconds)\n"\
        + "app::clock_pulse::interval                                    = 0.0 <#interval_seconds>\n\n"\
        + "app::scene::time_volume_ts::nb_time_steps                     = <#nb_time_steps>\n"\
        + "app::scene::time_volume_ts::interval                          = 0.0 <#interval_seconds>\n\n"\
        + "#------------------------------------------------------------\n"\
        + "### Volumes\n\n"\
        + "## individual settings for volume group 0\n"\
        + "app::scene::vector_field_s::size                              = <#spatial_dimensions>\n"\
        + "app::scene::vector_field_s::input_directory                   = <#volume_dir_path>\n"\
        + "app::scene::vector_field_s::input_file_base_name              = <#volume_name_s>\n"\
        + "app::scene::vector_field_s::input_file_extension              = .raw\n"\
        + "app::scene::vector_field_s::format                            = float32_4\n"\
        + "app::scene::vector_field_s::input_time_step_enum_stride       = <#time_step_enum_stride>\n"\
        + "app::scene::vector_field_s::input_time_step_enum_offset       = <#time_step_enum_offset>\n"\
        + "app::scene::vector_field_s::zyx_to_xyz                        = <#zyx_to_xyz>\n\n\n"\
        + "## individual settings for volume group 1\n"\
        + "app::scene::vector_field_v::size                              = <#spatial_dimensions>\n"\
        + "app::scene::vector_field_v::input_directory                   = <#volume_dir_path>\n"\
        + "app::scene::vector_field_v::input_file_base_name              = <#volume_name_v>\n"\
        + "app::scene::vector_field_v::input_file_extension              = .raw\n"\
        + "app::scene::vector_field_v::format                            = float32_4\n"\
        + "app::scene::vector_field_v::input_time_step_enum_stride       = <#time_step_enum_stride>\n"\
        + "app::scene::vector_field_v::input_time_step_enum_offset       = <#time_step_enum_offset>\n"\
        + "app::scene::vector_field_v::zyx_to_xyz                        = <#zyx_to_xyz>\n\n\n"\
        + "## individual settings for volume group 2\n"\
        + "app::scene::vector_field_w::size                              = <#spatial_dimensions>\n"\
        + "app::scene::vector_field_w::input_directory                   = <#volume_dir_path>\n"\
        + "app::scene::vector_field_w::input_file_base_name              = <#volume_name_w>\n"\
        + "app::scene::vector_field_w::input_file_extension              = .raw\n"\
        + "app::scene::vector_field_w::format                            = float32_4\n"\
        + "app::scene::vector_field_w::input_time_step_enum_stride       = <#time_step_enum_stride>\n"\
        + "app::scene::vector_field_w::input_time_step_enum_offset       = <#time_step_enum_offset>\n"\
        + "app::scene::vector_field_w::zyx_to_xyz                        = <#zyx_to_xyz>\n"


'''------------------------------------------------------------------------------------------------------------------------'''
'''--------- pre computing tensor product bezier surfaces to determine if PV lines can cross through a given area ---------'''
'''------------------------------------------------------------------------------------------------------------------------'''


@numba.njit
def get_may_contain_pv_lines(v_volume, w_volume, dimX, dimY, dimZ, h):

    s_out = np.zeros((dimX, dimY, dimZ))

    for x in range(dimX):
        for y in range(dimY):
            for z in range(dimZ):

                # all lower, upper bound combinations

                X = ((x-h, x), (x, x+h))
                Y = ((y-h, y), (y, y+h))
                Z = ((z-h, z), (z, z+h))
                combinations = [(xx, yy, zz) for xx in X for yy in Y for zz in Z]

                flag = False
                for (xx, yy, zz) in combinations:

                    # worst case all 8 combinations must be checked
                    # edge cases left out for simplicity, worst case:
                    # in case of boundary point (corner): 8 x 1 cube
                    # in case of boundary line (edge): 4 x 2 cubes
                    # in case of boundary surface:  2 x 4 cubes

                    xmin = xx[0]
                    ymin = yy[0]
                    zmin = zz[0]

                    xmax = xx[1]
                    ymax = yy[1]
                    zmax = zz[1]

                    result = voxel_may_contain_pv_wrapper(
                        v_volume,
                        w_volume,
                        x, y, z,
                        dimX, dimY, dimZ,
                        xmin, ymin, zmin,
                        xmax, ymax, zmax
                    )

                    # if one of the 8 subcubes has a potential
                    # pv line going through, we are done

                    if result:
                        flag = True
                        break

                # False if and only if all 8 subcubes
                # returned False, i.e. 100% certain no
                # PV line goes through any of the subcubes

                s_out[x, y, z] = flag

    return s_out


@numba.njit
def voxel_may_contain_pv_wrapper(
        v, w, x, y, z, dimX, dimY, dimZ, xmin, ymin, zmin, xmax, ymax, zmax):

    if (xmin < 0):
        xmin = x

    elif (xmax >= dimX):
        xmax = x

    if (ymin < 0):
        ymin = y

    elif (ymax >= dimY):
        ymax = y

    if (zmin < 0):
        zmin = z

    elif (zmax >= dimZ):
        zmax = z

    return voxel_may_contain_pv(

        v[xmin, ymin, zmin],
        v[xmin, ymin, zmax],
        v[xmin, ymax, zmin],
        v[xmin, ymax, zmax],
        v[xmax, ymin, zmin],
        v[xmax, ymin, zmax],
        v[xmax, ymax, zmin],
        v[xmax, ymax, zmax],

        w[xmin, ymin, zmin],
        w[xmin, ymin, zmax],
        w[xmin, ymax, zmin],
        w[xmin, ymax, zmax],
        w[xmax, ymin, zmin],
        w[xmax, ymin, zmax],
        w[xmax, ymax, zmin],
        w[xmax, ymax, zmax],

    )


@numba.njit
def voxel_may_contain_pv(
        v000, v001, v010, v011, v100, v101, v110, v111,
        w000, w001, w010, w011, w100, w101, w110, w111):

    return quad_may_contain_pv(v000, v001, v010, v011, w000, w001, w010, w011)\
        or quad_may_contain_pv(v100, v101, v110, v111, w100, w101, w110, w111)\
        or quad_may_contain_pv(v000, v001, v100, v101, w000, w001, w100, w101)\
        or quad_may_contain_pv(v010, v011, v110, v111, w010, w011, w110, w111)\
        or quad_may_contain_pv(v000, v010, v100, v110, w000, w010, w100, w110)\
        or quad_may_contain_pv(v001, v011, v101, v111, w001, w011, w101, w111)


@numba.njit
def quad_may_contain_pv(v00, v01, v10, v11, w00, w01, w10, w11):

    XY = ((1, 2), (2, 0), (0, 1))

    for (x, y) in XY:

        # Check whether the surface spanned in the first component has no zero crossings

        # bezier coefficients (bernstein polynomials)
        Z0 = (v00[x]*w00[y] - v00[y]*w00[x])
        Z1 = 0.5*(v00[x]*w01[y] + v01[x]*w00[y] - v00[y]*w01[x] - v01[y]*w00[x])
        Z2 = (v01[x]*w01[y] - v01[y]*w01[x])
        Z3 = 0.5*(v00[x]*w10[y] + v10[x]*w00[y] - v00[y]*w10[x] - v10[y]*w00[x])
        Z4 = 0.25*(v00[x]*w11[y] + v01[x]*w10[y] + v10[x]*w01[y] + v11[x]*w00[y]
            - v00[y]*w11[x] - v01[y]*w10[x] - v10[y]*w01[x] - v11[y]*w00[x])
        Z5 = 0.5*(v01[x]*w11[y] + v11[x]*w01[y] - v01[y]*w11[x] - v11[y]*w01[x])
        Z6 = (v10[x]*w10[y] - v10[y]*w10[x])
        Z7 = 0.5*(v10[x]*w11[y] + v11[x]*w10[y] - v10[y]*w11[x] - v11[y]*w10[x])
        Z8 = (v11[x]*w11[y] - v11[y]*w11[x])

        # use convex hull property to check, whether zero crossings exist for the scalar component
        are_all_bigger_zero = (Z0 > 0 and Z1 > 0 and Z2 > 0 and Z3 > 0 and Z4 > 0 and Z5 > 0 and Z6 > 0 and Z7 > 0 and Z8 > 0)
        are_all_smaller_zero = (Z0 < 0 and Z1 < 0 and Z2 < 0 and Z3 < 0 and Z4 < 0 and Z5 < 0 and Z6 < 0 and Z7 < 0 and Z8 < 0)
        if (are_all_bigger_zero or are_all_smaller_zero):
            return 0.0  # 100% certain there does not exist any zero crossing

    # any of the components could still have no zero crossings;
    return 1.0  # possibility of false positives


'''------------------------------------------------------------------------------------------------------------------------'''
'''-------------------------------------------- computing the filter volume s ---------------------------------------------'''
'''------------------------------------------------------------------------------------------------------------------------'''


def get_s_volume(PD):

    count = 0

    s_results = compute_s_entries(
        PD["dimX"],
        PD["dimY"],
        PD["dimZ"],
        PD["S_RADIUS"],
        PD["S_COLOR"],
        PD["S_ALPHA"],
        PD["j_numerical"],
        PD["j_numerical_eigenvalues"],
        PD["SWIRLING_STRENGTH_THRESH_LOW"],
        PD["VORTICITY_MAGNITUDE_THRESH_LOW"]
    )

    print("\nInitially, all filters have {} entries set to True".format(PD["dimX"]*PD["dimY"]*PD["dimZ"]))

    PD["s_a_no_line_filter"] = np.reshape(s_results[0], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
    count = print_filter_information(PD, count, PD["s_a_no_line_filter"][:, :, :, 0], "s_a_no_line_filter", "swirling_strength")

    PD["s_a_vortex_corelines"] = np.reshape(s_results[1], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
    count = print_filter_information(PD, count, PD["s_a_vortex_corelines"][:, :, :, 0], "s_a_vortex_corelines", "swirling_strength > 0 (mandatory)")

    PD["s_a_bifurcation_lines"] = np.reshape(s_results[2], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
    count = print_filter_information(PD, count, PD["s_a_bifurcation_lines"][:, :, :, 0], "s_a_bifurcation_lines", "swirling_strength <= 0 (mandatory)")

    PD["s_a_vorticity_extremal_lines"] = np.reshape(s_results[3], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
    count = print_filter_information(PD, count, PD["s_a_vorticity_extremal_lines"][:, :, :, 0], "s_a_vorticity_extremal_lines", "swirling_strength + all reals < 0")

    PD["s_b_no_line_filter"] = np.reshape(s_results[4], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
    count = print_filter_information(PD, count, PD["s_b_no_line_filter"][:, :, :, 0], "s_b_no_line_filter", "swirling_strength")

    PD["s_b_vortex_corelines"] = np.reshape(s_results[5], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
    count = print_filter_information(PD, count, PD["s_b_vortex_corelines"][:, :, :, 0], "s_b_vortex_corelines", "swirling_strength > 0 (mandatory)")

    PD["s_b_bifurcation_lines"] = np.reshape(s_results[6], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
    count = print_filter_information(PD, count, PD["s_b_bifurcation_lines"][:, :, :, 0], "s_b_bifurcation_lines", "swirling_strength <= 0 (mandatory)")

    PD["s_b_vorticity_extremal_lines"] = np.reshape(s_results[7], newshape=(PD["dimX"], PD["dimY"], PD["dimZ"], PD["dimC"]))
    count = print_filter_information(PD, count, PD["s_b_vorticity_extremal_lines"][:, :, :, 0], "s_b_vorticity_extremal_lines", "swirling_strength + all reals < 0")

    mask_a = np.ones((PD["dimX"], PD["dimY"], PD["dimZ"]), np.bool)

    if PD["TENSORPRODUCTS"]:
        mask_a = PD["a_pv_line_tensorprod_filter"]
        count = print_filter_information(PD, count, mask_a, "mask_a", "initialized by a_pv_line_tensorprod_filter")
        struct_ele = ndimage.generate_binary_structure(3, 3)  # 3x3x3 structuring element with all ones (all neighbours)
        mask_a = scipy.ndimage.morphology.binary_dilation(input=mask_a, structure=struct_ele, iterations=1, output=None, origin=0)
        count = print_filter_information(PD, count, mask_a, "mask_a", "binary morphology operators (dilatation, opening, closing, etc)")

    mask_b = np.ones((PD["dimX"], PD["dimY"], PD["dimZ"]), np.bool)

    if PD["TENSORPRODUCTS"]:
        mask_b = PD["b_pv_line_tensorprod_filter"]
        count = print_filter_information(PD, count, mask_b, "mask_b", "initialized by b_pv_line_tensorprod_filter")
        struct_ele = ndimage.generate_binary_structure(3, 3)  # 3x3x3 structuring element with all ones (all neighbours)
        mask_b = scipy.ndimage.morphology.binary_dilation(input=mask_b, structure=struct_ele, iterations=1, output=None, origin=0)
        count = print_filter_information(PD, count, mask_b, "mask_b", "binary morphology operators (dilatation, opening, closing, etc)")

    if "swirling-jet" in PD["VOLUME_NAME"]:
        swirling_jet_center_distances_indices_a = (PD["swirling_jet_center_distances"] > PD["SWIRLING_JET_CENTER_DISTANCE_A_THRESH_HIGH"]*202)
        swirling_jet_center_distances_indices_b = (PD["swirling_jet_center_distances"] > PD["SWIRLING_JET_CENTER_DISTANCE_B_THRESH_HIGH"]*202)
        mask_a[swirling_jet_center_distances_indices_a] = 0.0
        count = print_filter_information(PD, count, mask_a, "mask_a", "swirling_jet_center_distances_a")
        mask_b[swirling_jet_center_distances_indices_b] = 0.0
        count = print_filter_information(PD, count, mask_b, "mask_b", "swirling_jet_center_distances_b")

    # mask that affects both a and b
    mask = np.ones((PD["dimX"], PD["dimY"], PD["dimZ"]), dtype=np.float64)

    velocity_magnitude = np.linalg.norm(PD["v_numerical"], ord=None, axis=(3))
    velocity_magnitude_indices = (velocity_magnitude < PD["VELOCITY_MAGNITUDE_THRESH_LOW"])
    mask[velocity_magnitude_indices] = 0.0
    count = print_filter_information(PD, count, mask, "mask", "velocity_magnitude")

    vxa_magnitude = np.linalg.norm(PD["vxa_numerical"], ord=None, axis=(3))
    vxa_magnitude_indices = (vxa_magnitude > PD["VXW_MAGNITUDE_THRESH_HIGH"])
    mask_a[vxa_magnitude_indices] = 0.0
    count = print_filter_information(PD, count, mask_a, "mask_a", "vxa_magnitude")

    vxb_magnitude = np.linalg.norm(PD["vxb_numerical"], ord=None, axis=(3))
    vxb_magnitude_indices = (vxb_magnitude > PD["VXW_MAGNITUDE_THRESH_HIGH"])
    mask_b[vxb_magnitude_indices] = 0.0
    count = print_filter_information(PD, count, mask_b, "mask_b", "vxb_magnitude")

    mask_a = np.multiply(mask_a, mask)
    count = print_filter_information(PD, count, mask_a, "mask_a", "mask_a * mask")

    mask_b = np.multiply(mask_b, mask)
    count = print_filter_information(PD, count, mask_b, "mask_b", "mask_b * mask")

    filter_a = 1.0 - mask_a
    filter_a = filter_a.astype(np.bool)
    count = print_filter_information(PD, count, filter_a, "filter_a", "1.0 - mask_a")

    filter_b = 1.0 - mask_b
    filter_b = filter_b.astype(np.bool)
    count = print_filter_information(PD, count, filter_b, "filter_b", "1.0 - mask_b")

    PD["s_a_no_line_filter"][filter_a] = 0.0
    count = print_filter_information(PD, count, PD["s_a_no_line_filter"][:, :, :, 0], "s_a_no_line_filter", "filter_a")

    PD["s_a_vortex_corelines"][filter_a] = 0.0
    count = print_filter_information(PD, count, PD["s_a_vortex_corelines"][:, :, :, 0], "s_a_vortex_corelines", "filter_a")

    PD["s_a_bifurcation_lines"][filter_a] = 0.0
    count = print_filter_information(PD, count, PD["s_a_bifurcation_lines"][:, :, :, 0], "s_a_bifurcation_lines", "filter_a")

    PD["s_a_vorticity_extremal_lines"][filter_a] = 0.0
    count = print_filter_information(PD, count, PD["s_a_vorticity_extremal_lines"][:, :, :, 0], "s_a_vorticity_extremal_lines", "filter_a")

    PD["s_b_no_line_filter"][filter_b] = 0.0
    count = print_filter_information(PD, count, PD["s_b_no_line_filter"][:, :, :, 0], "s_b_no_line_filter", "filter_b")

    PD["s_b_vortex_corelines"][filter_b] = 0.0
    count = print_filter_information(PD, count, PD["s_b_vortex_corelines"][:, :, :, 0], "s_b_vortex_corelines", "filter_b")

    PD["s_b_bifurcation_lines"][filter_b] = 0.0
    count = print_filter_information(PD, count, PD["s_b_bifurcation_lines"][:, :, :, 0], "s_b_bifurcation_lines", "filter_b")

    PD["s_b_vorticity_extremal_lines"][filter_b] = 0.0
    count = print_filter_information(PD, count, PD["s_b_vorticity_extremal_lines"][:, :, :, 0], "s_b_vorticity_extremal_lines", "filter_b")

    PD["vorticity_magnitudes"] = s_results[8]
    analyze_single_vector_across_volume(
        PD,
        PD["vorticity_magnitudes"],
        "vorticity magnitude stats",
        "vorticity_magnitude_stats"
    )

    PD["imaginary_eigenvalues"] = s_results[9]
    analyze_single_vector_across_volume(
        PD,
        PD["imaginary_eigenvalues"],
        "abs largest imaginary eigval stats",
        "abs_largest_imaginary_eigval_stats"
    )

    analyze_single_vector_across_volume(
        PD,
        velocity_magnitude,
        "velocity magnitude stats",
        "velocity_magnitude_stats"
    )

    return PD


@numba.njit
def compute_s_entries(
        dimX,
        dimY,
        dimZ,
        s_radius,
        s_color,
        s_alpha,
        Jacobian,
        eigenvalues,
        SWIRLING_STRENGTH_THRESH_LOW,
        VORTICITY_MAGNITUDE_THRESH_LOW):

    s_a_no_line_filter = np.ones((dimX, dimY, dimZ, 4))
    s_a_vortex_corelines = np.ones((dimX, dimY, dimZ, 4))
    s_a_bifurcation_lines = np.ones((dimX, dimY, dimZ, 4))
    s_a_vorticity_extremal_lines = np.ones((dimX, dimY, dimZ, 4))

    s_b_no_line_filter = np.ones((dimX, dimY, dimZ, 4))
    s_b_vortex_corelines = np.ones((dimX, dimY, dimZ, 4))
    s_b_bifurcation_lines = np.ones((dimX, dimY, dimZ, 4))
    s_b_vorticity_extremal_lines = np.ones((dimX, dimY, dimZ, 4))

    vorticity_magnitudes = np.ones((dimX, dimY, dimZ))
    imaginary_eigenvalues = np.ones((dimX, dimY, dimZ))

    radius = np.ones((dimX, dimY, dimZ))
    color = np.ones((dimX, dimY, dimZ))
    alpha = np.ones((dimX, dimY, dimZ))

    for x in range(dimX):
        for y in range(dimY):
            for z in range(dimZ):

                Jxyz = Jacobian[x, y, z]
                vorticity_magn = math.sqrt((Jxyz[2, 1] - Jxyz[1, 2])**2 + (Jxyz[0, 2] - Jxyz[2, 0])**2 + (Jxyz[1, 0] - Jxyz[0, 1])**2)

                #eigenvalues_real = np.real(eigenvalues[x, y, z])
                eigenvalues_imag = np.imag(eigenvalues[x, y, z])
                eigenvalues_imag_max = np.max(eigenvalues_imag)

                vorticity_magnitudes[x, y, z] = vorticity_magn
                imaginary_eigenvalues[x, y, z] = eigenvalues_imag_max

                radius[x, y, z] = eigenvalues_imag_max
                color[x, y, z] = vorticity_magn
                alpha[x, y, z] = eigenvalues_imag_max

                # Refer to
                # https://docs.python.org/3.3/library/stdtypes.html?highlight=frozenset#boolean-values
                # "In numeric contexts (for example when used as the argument to an arithmetic operator),
                # they (Boolean Values) behave like the integers 0 and 1, respectively."

                # sujudi-haimes / focii + center / swirling
                # requires a complex-conjugate pair of eigenvalues
                has_complex_eigs = ( eigenvalues_imag_max != 0 )

                # complex-conjugate eigs are a necessary condition for swirling;
                # higher abs val => stronger swirling => keep areas that are above threshold
                is_swirling_strength_satisfied = ( eigenvalues_imag_max >= SWIRLING_STRENGTH_THRESH_LOW )

                # vorticity is also a good filter
                is_vorticity_magnitude_satisfied = ( vorticity_magn >= VORTICITY_MAGNITUDE_THRESH_LOW )

                # peikert-roth / nodes + saddle / bifurcation requires
                # in the plane around the pv line to be: 1. real 2. different signs 3. nnz

                # fill up filters

                s_a_no_line_filter[x, y, z, 0] = s_a_no_line_filter[x, y, z, 0] * is_swirling_strength_satisfied * is_vorticity_magnitude_satisfied
                s_b_no_line_filter[x, y, z, 0] = s_b_no_line_filter[x, y, z, 0] * is_swirling_strength_satisfied * is_vorticity_magnitude_satisfied

                s_a_vortex_corelines[x, y, z, 0] = s_a_vortex_corelines[x, y, z, 0] * is_swirling_strength_satisfied * is_vorticity_magnitude_satisfied * has_complex_eigs
                s_b_vortex_corelines[x, y, z, 0] = s_b_vortex_corelines[x, y, z, 0] * is_swirling_strength_satisfied * is_vorticity_magnitude_satisfied * has_complex_eigs


    # Rescale / Normalize transfer functions to a [0, 1] range
    #radius = np.divide(radius - np.min(radius), np.max(radius) - np.min(radius))
    #color = np.divide(color - np.min(color), np.max(color) - np.min(color))
    #alpha = np.divide(alpha - np.min(alpha), np.max(alpha) - np.min(alpha))

    s_a_no_line_filter[:, :, :, 1] = radius
    s_a_no_line_filter[:, :, :, 2] = color
    s_a_no_line_filter[:, :, :, 3] = alpha

    s_a_vortex_corelines[:, :, :, 1] = radius
    s_a_vortex_corelines[:, :, :, 2] = color
    s_a_vortex_corelines[:, :, :, 3] = alpha

    s_a_bifurcation_lines[:, :, :, 1] = radius
    s_a_bifurcation_lines[:, :, :, 2] = color
    s_a_bifurcation_lines[:, :, :, 3] = alpha

    s_a_vorticity_extremal_lines[:, :, :, 1] = radius
    s_a_vorticity_extremal_lines[:, :, :, 2] = color
    s_a_vorticity_extremal_lines[:, :, :, 3] = alpha

    s_b_no_line_filter[:, :, :, 1] = radius
    s_b_no_line_filter[:, :, :, 2] = color
    s_b_no_line_filter[:, :, :, 3] = alpha

    s_b_vortex_corelines[:, :, :, 1] = radius
    s_b_vortex_corelines[:, :, :, 2] = color
    s_b_vortex_corelines[:, :, :, 3] = alpha

    s_b_bifurcation_lines[:, :, :, 1] = radius
    s_b_bifurcation_lines[:, :, :, 2] = color
    s_b_bifurcation_lines[:, :, :, 3] = alpha

    s_b_vorticity_extremal_lines[:, :, :, 1] = radius
    s_b_vorticity_extremal_lines[:, :, :, 2] = color
    s_b_vorticity_extremal_lines[:, :, :, 3] = alpha

    return (
        s_a_no_line_filter,
        s_a_vortex_corelines,
        s_a_bifurcation_lines,
        s_a_vorticity_extremal_lines,
        s_b_no_line_filter,
        s_b_vortex_corelines,
        s_b_bifurcation_lines,
        s_b_vorticity_extremal_lines,
        vorticity_magnitudes,
        imaginary_eigenvalues
    )
