# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
# ==============================================================================
import sys
import multiprocessing as mp
import os
import tempfile
import traceback
from qti.aisw.converters.common.utils.converter_utils import log_info, log_warning, log_error


dlc_import_error = False
try:
    from qti.aisw.converters.common import modeltools
except ImportError:
    try:
        # May need this import for QAIRT
        from qti.aisw.dlc_utils import modeltools
    except ImportError:
        dlc_import_error = True
        log_info("Unable to import DLC utilties")

from qti.aisw.converters.common import ir_graph as ir_graph_lib
IrGraph = ir_graph_lib.IrGraph


def prepare_parent_environment_for_aimet(init_method):
    def wrapper(*args, **kwargs):
        init_method(*args, **kwargs)

        aimet_venv_python_path = args[0].aimet_venv_python_path
        aimet_venv_base_dir = os.path.dirname(os.path.dirname(aimet_venv_python_path))

        if sys.platform == "win32":
            site_packages = os.path.join(aimet_venv_base_dir, sys.platlibdir, "site-packages")
        else:
            major_v = sys.version_info.major
            minor_v = sys.version_info.minor
            site_packages = os.path.join(aimet_venv_base_dir, sys.platlibdir, f"python{major_v}.{minor_v}", "site-packages")

        import site
        site.addsitedir(site_packages)

        # Order the python path so that the site-package of the virtual environment is at the beginning
        # of the python path list. This prioritizes the version of a package in the virtual environment
        sys.path = ([path for path in sys.path if aimet_venv_base_dir in path] +
                    [path for path in sys.path if aimet_venv_base_dir not in path])

    return wrapper


def restore_parent_environment_after_aimet(terminate_method):
    def wrapper(*args, **kwargs):
        terminate_method(*args, **kwargs)

        # Restore sys.path, after terminating AIMET process
        sys.path = args[0].parent_sys_path

    return wrapper


def serialize_ir_graph_to_dlc(ir_graph, path: str, filename: str):
    """
    Serialize IrGraph to dlc and save it to the specified path with the provided filename
    :param ir_graph: IrGraph to be serialized
    :param path: Path to save exported dlc model
    :param filename: filename to save exported dlc model
    """
    if not dlc_import_error:
        _serialize_ir_graph_to_dlc(ir_graph, path, filename)
    else:
        raise Exception('Unable to serialize IR graph to DLC')


def _serialize_ir_graph_to_dlc(ir_graph, path: str, filename: str):
    """
    Serialize IrGraph to dlc and save it to the specified path with the provided filename
    :param ir_graph: IrGraph to be serialized
    :param path: Path to save exported dlc model
    :param filename: filename to save exported dlc model
    """
    dlc_serializer = modeltools.IrDlcSerializer(os.path.join(path, filename + ".dlc"))
    dlc_serializer.initialize()
    dlc_serializer.serialize(ir_graph)
    dlc_serializer.finish()


def get_ir_graph_from_dlc(dlc_path: str):
    """
    Obtain IR Graph from DLC (non quantized).
    :param dlc_path: Path where dlc is located
    """
    if not dlc_import_error:
        dlc_reader_obj = _get_dlc_reader(dlc_path)
        ir_graph = dlc_reader_obj.get_ir_graph()
        return ir_graph, dlc_reader_obj
    else:
        raise Exception('Unable to obtain IR graph from dlc as relevant utils are not imported')


def get_dlc_reader(dlc_path: str):
    if not dlc_import_error:
        dlc_reader_obj = _get_dlc_reader(dlc_path)
        return dlc_reader_obj
    else:
        raise Exception('Unable to obtain IR graph from dlc as relevant utils are not imported')


def _get_dlc_reader(dlc_path: str):
    """
    Obtain IR Graph from DLC (non quantized).
    :param dlc_path: Path where dlc is located
    """
    dlc_reader = modeltools.IrDlcReader()
    dlc_reader.open(dlc_path)
    return dlc_reader


def get_python_executable():
    aimet_env_python_exec = os.environ.get("AIMET_ENV_PYTHON")
    error_msg = ('Provided python executable at $AIMET_ENV_PYTHON is invalid. Please run '
                 'aimet_env_setup.sh to ensure AIMET_ENV_PYTHON is set to <aimet_venv>/lib/python')
    if os.path.exists(aimet_env_python_exec):
        # This returns python version, must contain 'python' in version string,
        # if it is a valid python interpreter
        try:
            python_version = os.popen(f'{aimet_env_python_exec} --version').read().strip()
            assert 'python' in python_version.lower(), error_msg
            log_info('Validated environment variable, AIMET_ENV_PYTHON')
            return aimet_env_python_exec
        except Exception:
            raise EnvironmentError(error_msg)
    else:
        raise EnvironmentError(error_msg)


def quantize_model_with_aimet(dlc_path, conn, tmpdir, opts):
    """
    Call this function within a subprocess to execute aimet specific code in a separate virtual environment
    """
    quantized_dlc_path = None
    try:
        # Import this only after adding the virtual environment's site package to python path
        from qti.aisw.converters.aimet.qnn_quantsim_adapter import QnnToAimetAdapter
        ir_graph, dlc_reader = get_ir_graph_from_dlc(dlc_path)
        qnn_adapter = QnnToAimetAdapter(ir_graph, opts, datadir=tmpdir, use_cuda=True)
        if qnn_adapter.is_valid_opts():
            # Persist DLC reader to avoid DLC shared memory being freed. Read IR graph from DLC reader
            quantized_dlc_path = qnn_adapter.generate_quantized_dlc(dlc_path)
    except Exception as e:
        traceback.print_exc()
    finally:
        conn.send([quantized_dlc_path,])
        conn.close()


class AimetProcess(mp.Process):
    @prepare_parent_environment_for_aimet
    def __init__(self, target, args, aimet_venv_python_path, parent_sys_path):
        super(AimetProcess, self).__init__(target=target, args=args)
        self.target = target
        self.args = args
        self.aimet_venv_python_path = aimet_venv_python_path
        self.parent_sys_path = parent_sys_path

    def run(self):
        self.target(*self.args)

    @restore_parent_environment_after_aimet
    def terminate(self):
        super(AimetProcess, self).terminate()


def aimet_quantizer(ir_graph, opts):
    aimet_env_python_exec = get_python_executable()
    if not aimet_env_python_exec:
        raise EnvironmentError(
            """Environment variable 'AIMET_ENV_PYTHON' not set.
            Please run  'source $QNN_SRC/QTI/scripts/aimet_env_setup.sh --env-path <PATH>' if you want to use aimet quantizer
            or omit the '--use_aimet_quantizer' flag to use the default quantizer"""
        )
    # Create a multiprocessing context with start method 'spawn' and set the python executable path
    with tempfile.TemporaryDirectory() as tmpdir:
        unquantized_dlc_filename = 'model_fp'
        unquantized_dlc_path = tmpdir
        serialize_ir_graph_to_dlc(ir_graph, unquantized_dlc_path, unquantized_dlc_filename)
        # Set multiprocessing context to 'spawn' to keep AIMET process be independent of main process environment
        mp.set_start_method("spawn", force=True)
        mp.set_executable(aimet_env_python_exec)
        # Create a process and run aimet-specific code within the context of that process
        parent_conn, child_conn = mp.Pipe()
        fullpath = os.path.join(unquantized_dlc_path, unquantized_dlc_filename + '.dlc')
        process = AimetProcess(target=quantize_model_with_aimet,
                               args=(fullpath, child_conn, tmpdir, opts),
                               aimet_venv_python_path=aimet_env_python_exec,
                               parent_sys_path=sys.path.copy())
        process.start()
        retval = parent_conn.recv()
        quantized_dlc_path = retval[0]
        process.join()
        process.terminate()
        if quantized_dlc_path is not None and os.path.exists(quantized_dlc_path):
            reader = get_dlc_reader(quantized_dlc_path)
            return reader
        else:
            log_error('Exception occured in Spawned AIMET Process, Unable to proceed with Quantization')
            sys.exit()


class AimetQuantizerOpts:
    def __init__(self,
                 input_network,
                 output_path,
                 input_list,
                 quant_schemes,
                 float_fallback,
                 disable_legacy_quant_scheme_opts,
                 algorithms,
                 act_bitwidth,
                 weights_bitwidth,
                 bias_bitwidth,
                 float_bias_bw,
                 percentile_calibration_value,
                 ignore_encodings,
                 use_per_channel_quantization,
                 use_per_row_quantization,
                 use_native_input_files,
                 use_native_output_files,
                 config=None):

        # TODO: Remove this once Windows based AIMET build is available
        if sys.platform == "win32":
            raise RuntimeError('AIMETQuantizer is not supported on Windows yet')

        self.input_network = input_network
        self.output_path = output_path
        self.input_list = input_list
        self.quant_schemes = quant_schemes
        self.float_fallback = float_fallback
        # Flag to detect whether to use --act_quantizer and --param_quantizer (or) [--act_quantizer_calibration, --act_quantizer_schema]
        # [--param_quantizer_calibration, --param_quantizer_schema] to resolve AIMET Quant Scheme
        self.disable_legacy_quant_scheme_opts = disable_legacy_quant_scheme_opts
        self.algorithms = algorithms
        self.act_bitwidth = act_bitwidth
        self.weights_bitwidth = weights_bitwidth
        self.bias_bitwidth = bias_bitwidth
        self.float_bias_bw = float_bias_bw
        self.percentile_calibration_value = percentile_calibration_value
        self.ignore_encodings = ignore_encodings
        self.use_per_channel_quantization = use_per_channel_quantization
        self.use_per_row_quantization = use_per_row_quantization
        self.use_native_input_files = use_native_input_files
        self.use_native_output_files = use_native_output_files
        self.config = config
        self.validate_aimet_quant_opts()

    def validate_aimet_quant_opts(self):
        # TODO: Support --use_native_output_files if required
        if self.use_native_output_files:
            raise Exception("AIMET Quantizer doesn't support --use_native_output_files")

        # TODO: Support --bias_bitwidth 8
        if self.bias_bitwidth != 32:
            # TODO: raise Exception once the default is changed to 32
            log_warning(f"AIMET Quantizer doesn't support {self.bias_bitwidth} for --bias_bitwidth or --bias_bw, using 32")
            self.bias_bitwidth = 32

        if self.config is not None and self.float_fallback:
            raise Exception("Can't provide --config and --float_fallback together")

        if len(self.algorithms) != 0 and self.float_fallback:
            raise Exception("Can't provide --algorithms and --float_fallback together")

        if len(self.algorithms) > 1:
            raise RuntimeError("Currently AIMET Quantizer can't run more than one algorithm!")

        if self.config is not None:
            # When config is provided, algorithms should be provided and can take values 'amp', 'adaround' or 'autoquant'
            if (len(self.algorithms) == 1 and self.algorithms[0] not in ['amp', 'adaround', 'autoquant']) or len(self.algorithms) == 0:
                raise Exception("When --config is provided, --algorithms should be one of 'amp', 'adaround' or 'autoquant'")
        else:
            # When no config is provided, algorithms can take 'cle' or 'adaround' (default case)
            if len(self.algorithms) == 1 and self.algorithms[0] not in ['adaround', 'cle']:
                raise Exception("When no --config is provided, --algorithms can only take 'cle' or 'adaround'")


def aimet_dlc_quantizer(opts):
    aimet_env_python_exec = get_python_executable()
    if not aimet_env_python_exec:
        raise EnvironmentError(
            """Environment variable 'AIMET_ENV_PYTHON' not set.
            Please run  'source $QNN_SRC/QTI/scripts/aimet_env_setup.sh --env-path <PATH>' if you want to use aimet quantizer
            or omit the '--use_aimet_quantizer' flag to use the default quantizer"""
        )
    # Create a multiprocessing context with start method 'spawn' and set the python executable path
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set multiprocessing context to 'spawn' to keep AIMET process be independent of main process environment
        mp.set_start_method("spawn", force=True)
        mp.set_executable(aimet_env_python_exec)
        # Create a process and run aimet-specific code within the context of that process
        parent_conn, child_conn = mp.Pipe()
        process = AimetProcess(target=quantize_model_with_aimet,
                               args=(opts.input_network, child_conn, tmpdir, opts),
                               aimet_venv_python_path=aimet_env_python_exec,
                               parent_sys_path=sys.path.copy())
        process.start()
        retval = parent_conn.recv()
        quantized_dlc_path = retval[0]
        process.join()
        process.terminate()
        if quantized_dlc_path is not None and os.path.exists(quantized_dlc_path):
            reader = get_dlc_reader(quantized_dlc_path)
            return reader
        else:
            log_error('Exception occured in Spawned AIMET Process, Unable to proceed with Quantization')
            sys.exit()