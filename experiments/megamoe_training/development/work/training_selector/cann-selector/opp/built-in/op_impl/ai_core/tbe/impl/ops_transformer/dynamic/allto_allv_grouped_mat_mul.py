#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
"""

import re
import os, sys
import ctypes
import json
import shutil


from asc_op_compile_base.common.platform import get_soc_spec
from asc_op_compile_base.common.utils import para_check
from asc_op_compile_base.asc_op_compiler import compile_op, replay_op, check_op_cap, generalize_op_params, get_code_channel, OpInfo
from asc_op_compile_base.asc_op_compiler.compile_op import CommonUtility, AscendCLogLevel
from asc_op_compile_base.common.buildcfg import get_default_build_config
from asc_op_compile_base.common.buildcfg import get_current_build_config
from asc_op_compile_base.common import register as tbe_register
__version__ = '2.0.0'


PYF_PATH = os.path.dirname(os.path.realpath(__file__))

DTYPE_MAP = {"float32": ["DT_FLOAT", "float"],
    "float16": ["DT_FLOAT16", "half"],
    "int8": ["DT_INT8", "int8_t"],
    "int16": ["DT_INT16", "int16_t"],
    "int32": ["DT_INT32", "int32_t"],
    "int64": ["DT_INT64", "int64_t"],
    "uint1": ["DT_UINT1", "uint1b_t"],
    "uint8": ["DT_UINT8", "uint8_t"],
    "uint16": ["DT_UINT16", "uint16_t"],
    "uint32": ["DT_UINT32", "uint32_t"],
    "uint64": ["DT_UINT64", "uint64_t"],
    "bool": ["DT_BOOL", "bool"],
    "double": ["DT_DOUBLE", "double"],
    "dual": ["DT_DUAL", "unknown"],
    "dual_sub_int8": ["DT_DUAL_SUB_INT8", "unknown"],
    "dual_sub_uint8": ["DT_DUAL_SUB_UINT8", "unknown"],
    "string": ["DT_STRING", "unknown"],
    "complex32": ["DT_COMPLEX32", "complex32"],
    "complex64": ["DT_COMPLEX64", "complex64"],
    "complex128": ["DT_COMPLEX128", "unknown"],
    "qint8": ["DT_QINT8", "unknown"],
    "qint16": ["DT_QINT16", "unknown"],
    "qint32": ["DT_QINT32", "unknown"],
    "quint8": ["DT_QUINT8", "unknown"],
    "quint16": ["DT_QUINT16", "unknown"],
    "resource": ["DT_RESOURCE", "unknown"],
    "string_ref": ["DT_STRING_REF", "unknown"],
    "int4": ["DT_INT4", "int4b_t"],
    "bfloat16": ["DT_BF16", "bfloat16_t"],
    "float8_e5m2": ["DT_FLOAT8_E5M2", "fp8_e5m2_t"],
    "float8_e4m3fn": ["DT_FLOAT8_E4M3FN", "fp8_e4m3fn_t"],
    "hifloat8":["DT_HIFLOAT8", "hifloat8_t"],
    "float8_e8m0":["DT_FLOAT8_E8M0", "fp8_e8m0_t"],
    "float4_e2m1":["DT_FLOAT4_E2M1", "fp4x2_e2m1_t"],
    "float4_e1m2":["DT_FLOAT4_E1M2", "fp4x2_e1m2_t"],
    "int2": ["DT_INT2", "int2b_t"]}

def add_dtype_fmt_option_single(x, x_n, is_ref: bool = False):
    options = []
    x_fmt = x.get("format")
    x_dtype = x.get("dtype")
    x_n_in_kernel = x_n + '_REF' if is_ref else x_n
    options.append("-DDTYPE_{n}={t}".format(n=x_n_in_kernel, t=DTYPE_MAP.get(x_dtype)[1]))
    options.append("-DORIG_DTYPE_{n}={ot}".format(n=x_n_in_kernel, ot=DTYPE_MAP.get(x_dtype)[0]))
    options.append("-DFORMAT_{n}=FORMAT_{f}".format(n=x_n_in_kernel, f=x_fmt))
    return options

def get_dtype_fmt_options(__inputs__, __outputs__):
    options = []
    input_names = ['gmm_x', 'gmm_weight', 'send_counts_tensor', 'recv_counts_tensor', 'mm_x', 'mm_weight']
    output_names = ['gmm_y', 'mm_y', 'permute_out']
    unique_param_name_set = set()
    for idx, x in enumerate(__inputs__):
        if x is None:
            continue
        x_n = input_names[idx].upper()
        unique_param_name_set.add(x_n)
        options += add_dtype_fmt_option_single(x, x_n)

    for idx, x in enumerate(__outputs__):
        if x is None:
            continue
        x_n = output_names[idx].upper()
        if x_n in unique_param_name_set:
            options += add_dtype_fmt_option_single(x, x_n, True)
        else:
            options += add_dtype_fmt_option_single(x, x_n)
    return options

def load_dso(so_path):
    try:
        ctypes.CDLL(so_path)
    except OSError as error :
        CommonUtility.print_compile_log("", error, AscendCLogLevel.LOG_ERROR)
        raise RuntimeError("cannot open %s" %(so_path))
    else:
        msg = "load so succ " + so_path
        CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)

def get_shortsoc_compile_option(compile_option_list: list, shortsoc:str):
    compile_options = []
    if shortsoc in compile_option_list:
        compile_options.extend(compile_option_list[shortsoc])
    if '__ALLSOC__' in compile_option_list:
        compile_options.extend(compile_option_list['__ALLSOC__'])
    return compile_options

def get_kernel_source(src_file, dir_snake, dir_ex):
    src = os.path.join(PYF_PATH, "op_kernel", src_file)
    if os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, "..", "ascendc", dir_snake, "op_kernel", src_file)
    if os.path.exists(src):
        return src
    src_ex = os.path.join(PYF_PATH, "..", "ascendc", dir_ex, "op_kernel", src_file)
    if os.path.exists(src_ex):
        return src_ex
    src_ex = os.path.join(PYF_PATH, "..", "ascendc", dir_ex, src_file)
    if os.path.exists(src_ex):
        return src_ex
    src = os.environ.get('BUILD_KERNEL_SRC')
    if src and os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, "..", "ascendc", dir_snake, src_file)
    if os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, src_file)
    if os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, "..", "ascendc", dir_snake, dir_snake + ".cpp")
    if os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, "..", "ascendc", dir_ex, dir_ex + ".cpp")
    if os.path.exists(src):
        return src
    src = os.path.join(PYF_PATH, "..", "ascendc", os.path.splitext(src_file)[0], src_file)
    if os.path.exists(src):
        return src
    return src_ex

def _build_args(gmm_x_in__, gmm_weight_in__, send_counts_tensor_in__, recv_counts_tensor_in__, mm_x_in__, mm_weight_in__, gmm_y_out_, mm_y_out_, permute_out_out_, group, ep_world_size, send_counts, recv_counts, trans_gmm_weight, trans_mm_weight, permute_out_flag, comm_mode):
    __inputs__ = []
    for arg in [gmm_x_in__, gmm_weight_in__, send_counts_tensor_in__, recv_counts_tensor_in__, mm_x_in__, mm_weight_in__]:
        if arg is not None:
            if isinstance(arg, (list, tuple)):
                if len(arg) == 0:
                    continue
                __inputs__.append(arg[0])
            else:
                __inputs__.append(arg)
        else:
            __inputs__.append(arg)
    __outputs__ = []
    for arg in [gmm_y_out_, mm_y_out_, permute_out_out_]:
        if arg is not None:
            if isinstance(arg, (list, tuple)):
                if len(arg) == 0:
                    continue
                __outputs__.append(arg[0])
            else:
                __outputs__.append(arg)
        else:
            __outputs__.append(arg)
    __attrs__ = []
    if group is not None:
        attr = {}
        attr["name"] = "group"
        attr["dtype"] = "str"
        attr["value"] = group
        __attrs__.append(attr)
    if ep_world_size is not None:
        attr = {}
        attr["name"] = "ep_world_size"
        attr["dtype"] = "int"
        attr["value"] = ep_world_size
        __attrs__.append(attr)
    if send_counts is not None:
        attr = {}
        attr["name"] = "send_counts"
        attr["dtype"] = "list_int"
        attr["value"] = send_counts
        __attrs__.append(attr)
    if recv_counts is not None:
        attr = {}
        attr["name"] = "recv_counts"
        attr["dtype"] = "list_int"
        attr["value"] = recv_counts
        __attrs__.append(attr)
    if trans_gmm_weight is not None:
        attr = {}
        attr["name"] = "trans_gmm_weight"
        attr["dtype"] = "bool"
        attr["value"] = trans_gmm_weight
        __attrs__.append(attr)
    if trans_mm_weight is not None:
        attr = {}
        attr["name"] = "trans_mm_weight"
        attr["dtype"] = "bool"
        attr["value"] = trans_mm_weight
        __attrs__.append(attr)
    if permute_out_flag is not None:
        attr = {}
        attr["name"] = "permute_out_flag"
        attr["dtype"] = "bool"
        attr["value"] = permute_out_flag
        __attrs__.append(attr)
    if comm_mode is not None:
        attr = {}
        attr["name"] = "comm_mode"
        attr["dtype"] = "str"
        attr["value"] = comm_mode
        __attrs__.append(attr)
    return __inputs__, __outputs__, __attrs__

@tbe_register.register_operator("AlltoAllvGroupedMatMul", trans_bool_to_s8=False)
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_OUTPUT, para_check.OPTION_ATTR_STR, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_STR, para_check.KERNEL_NAME)
def allto_allv_grouped_mat_mul(gmm_x_in__, gmm_weight_in__, send_counts_tensor_in__=None, recv_counts_tensor_in__=None, mm_x_in__=None, mm_weight_in__=None, gmm_y_out_=None, mm_y_out_=None, permute_out_out_=None, group="", ep_world_size=0, send_counts=[], recv_counts=[], trans_gmm_weight=False, trans_mm_weight=False, permute_out_flag=False, comm_mode="ai_cpu", kernel_name="allto_allv_grouped_mat_mul", impl_mode = ""):
    # do ascendc build step
    if get_current_build_config("enable_op_prebuild"):
        return
    __inputs__, __outputs__, __attrs__ = _build_args(gmm_x_in__, gmm_weight_in__, send_counts_tensor_in__, recv_counts_tensor_in__, mm_x_in__, mm_weight_in__, gmm_y_out_, mm_y_out_, permute_out_out_, group, ep_world_size, send_counts, recv_counts, trans_gmm_weight, trans_mm_weight, permute_out_flag, comm_mode)
    options = get_dtype_fmt_options(__inputs__, __outputs__)
    options += ["-x", "cce"]
    bisheng = os.environ.get('BISHENG_REAL_PATH')
    if bisheng is None:
        bisheng = shutil.which("bisheng")
    if bisheng is not None:
        bisheng_path = os.path.dirname(bisheng)
        tikcpp_path = os.path.realpath(os.path.join(bisheng_path, "..", "..", "tikcpp"))
    else:
        tikcpp_path = os.path.realpath("/usr/local/Ascend/latest/compiler/tikcpp")
    options.append("-I" + os.path.join(PYF_PATH, "..", "ascendc", "common", "catlass", "include"))
    options.append("-I" + os.path.join(PYF_PATH, "..", "ascendc", "common", "tensor_api", "include"))
    options.append("-I" + os.path.join(PYF_PATH, "..", "ascendc", "common", "tensor_api"))
    options.append("-I" + tikcpp_path)
    options.append("-I" + os.path.join(tikcpp_path, "..", "..", "include"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "impl"))
    options.append("-I" + os.path.join(tikcpp_path, "tikcfw", "interface"))
    options.append("-I" + os.path.join(tikcpp_path, "..", "ascendc", "act"))
    options.append("-I" + os.path.join(PYF_PATH, "..", "ascendc", "common"))
    options.append("-I" + os.path.join(PYF_PATH, "..", "ascendc", "common", "op_kernel"))
    toolkit_path = os.environ.get('ASCEND_HOME_PATH')
    if toolkit_path is None:
        toolkit_path = os.path.realpath("/usr/local/Ascend/latest/")
    options.append("-I" + toolkit_path + os.path.join("/", os.uname().machine +"-linux", "asc", "atcos"))
    op_common_path = os.path.realpath(toolkit_path + "/include/op_common/")
    options.append("-I" + op_common_path)
    op_common_path = os.path.realpath(toolkit_path + "/pkg_inc/op_common/")
    options.append("-I" + op_common_path)
    if "impl_mode" in locals():
        if impl_mode == "high_performance":
            options.append("-DHIGH_PERFORMANCE=1")
        elif impl_mode == "high_precision":
            options.append("-DHIGH_PRECISION=1")
        elif "high_precision" in impl_mode and "high_performance" in impl_mode:
            options.append("-DHIGH_PRECISION=1 -DHIGH_PERFORMANCE=1")
    if get_current_build_config("enable_deterministic_mode") == 1:
        options.append("-DDETERMINISTIC_MODE=1")
    else:
        options.append("-DDETERMINISTIC_MODE=0")
    ascendc_api_version_header_path = os.path.join(tikcpp_path, "tikcfw/lib/ascendc_api_version.h")
    if os.path.exists(ascendc_api_version_header_path):
        with open(ascendc_api_version_header_path, "r") as ascendc_api_version_file:
            ascendc_api_version = re.findall(r"#define ASCENDC_API_VERSION (\d+)", ascendc_api_version_file.read())
            if ascendc_api_version:
                options.append(f"-DASCENDC_API_VERSION={ascendc_api_version[0]}")
    custom_compile_options = {'__ALLSOC__': ['--cce-auto-sync=off', '-mllvm', '-cce-aicore-dcci-before-kernel-end=false']},
    custom_all_compile_options = {'__ALLSOC__': ['-DNOT_DYNAMIC_COMPILE']},
    soc_version = get_soc_spec("SOC_VERSION")
    soc_short = get_soc_spec("SHORT_SOC_VERSION").lower()
    custom_compile_options_soc = get_shortsoc_compile_option(custom_compile_options[0], soc_short)
    custom_all_compile_options_soc = get_shortsoc_compile_option(custom_all_compile_options[0], soc_short)
    options += custom_all_compile_options_soc
    options += custom_compile_options_soc

    origin_func_name = "allto_allv_grouped_mat_mul"
    ascendc_src_dir_ex = "allto_allv_grouped_mat_mul"
    ascendc_src_dir = "allto_allv_grouped_mat_mul"
    ascendc_src_file = "allto_allv_grouped_mat_mul.cpp"
    src = get_kernel_source(ascendc_src_file, ascendc_src_dir, ascendc_src_dir_ex)

    msg = "start compile Ascend C Operator AlltoAllvGroupedMatMul, kernel name is " + kernel_name
    CommonUtility.print_compile_log("", msg, AscendCLogLevel.LOG_INFO)
    op_type = "AlltoAllvGroupedMatMul"
    code_channel = get_code_channel(src, kernel_name, op_type, options)
    op_info = OpInfo(kernel_name = kernel_name, op_type = op_type, inputs = __inputs__, outputs = __outputs__,\
        attrs = __attrs__ , impl_mode = impl_mode, origin_inputs=[gmm_x_in__, gmm_weight_in__, send_counts_tensor_in__, recv_counts_tensor_in__, mm_x_in__, mm_weight_in__], origin_outputs = [gmm_y_out_, mm_y_out_, permute_out_out_],\
                param_type_dynamic = False, mc2_ctx = ["mc2_context_0"], param_type_list = ['required', 'required', 'optional', 'optional', 'optional', 'optional', 'required', 'optional', 'optional'], init_value_list = [None, None, None],\
                output_shape_depend_on_compute = [])
    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})

def op_select_format(gmm_x_in__, gmm_weight_in__, send_counts_tensor_in__=None, recv_counts_tensor_in__=None, mm_x_in__=None, mm_weight_in__=None, gmm_y_out_=None, mm_y_out_=None, permute_out_out_=None, group="", ep_world_size=0, send_counts=[], recv_counts=[], trans_gmm_weight=False, trans_mm_weight=False, permute_out_flag=False, comm_mode="ai_cpu", impl_mode = ""):
    __inputs__, __outputs__, __attrs__ = _build_args(gmm_x_in__, gmm_weight_in__, send_counts_tensor_in__, recv_counts_tensor_in__, mm_x_in__, mm_weight_in__, gmm_y_out_, mm_y_out_, permute_out_out_, group, ep_world_size, send_counts, recv_counts, trans_gmm_weight, trans_mm_weight, permute_out_flag, comm_mode)
    result = check_op_cap("op_select_format", "AlltoAllvGroupedMatMul", __inputs__, __outputs__, __attrs__)
    return result.decode("utf-8")

def get_op_specific_info(gmm_x_in__, gmm_weight_in__, send_counts_tensor_in__=None, recv_counts_tensor_in__=None, mm_x_in__=None, mm_weight_in__=None, gmm_y_out_=None, mm_y_out_=None, permute_out_out_=None, group="", ep_world_size=0, send_counts=[], recv_counts=[], trans_gmm_weight=False, trans_mm_weight=False, permute_out_flag=False, comm_mode="ai_cpu", impl_mode = ""):
    __inputs__, __outputs__, __attrs__ = _build_args(gmm_x_in__, gmm_weight_in__, send_counts_tensor_in__, recv_counts_tensor_in__, mm_x_in__, mm_weight_in__, gmm_y_out_, mm_y_out_, permute_out_out_, group, ep_world_size, send_counts, recv_counts, trans_gmm_weight, trans_mm_weight, permute_out_flag, comm_mode)
    result = check_op_cap("get_op_specific_info", "AlltoAllvGroupedMatMul", __inputs__, __outputs__, __attrs__)
    return result.decode("utf-8")
