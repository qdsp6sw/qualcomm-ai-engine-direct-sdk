# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
from qti.aisw.accuracy_debugger.lib.inference_engine import inference_engine_repository
from qti.aisw.accuracy_debugger.lib.inference_engine.executors.nd_executor import Executor
from qti.aisw.accuracy_debugger.lib.utils.nd_constants import ComponentType, Framework, Engine


@inference_engine_repository.register(cls_type=ComponentType.executor, framework=None,
                                      engine=Engine.SNPE, engine_version="1.22.2.233")
class SNPEExecutor(Executor):

    def __init__(self, context):
        super(SNPEExecutor, self).__init__(context)
        self.executable = context.executable
        # Windows executable are differentiated by having '.exe' in the name.
        if context.architecture in ['x86_64-windows-msvc', 'wos-remote', 'wos']:
            self.executable = context.windows_executable

        self.container = context.arguments["container"]
        self.input_list = context.arguments["input_list"]
        self.runtime = 'dsp' if 'dsp' in context.runtime else context.runtime
        self.runtime_flag = context.arguments["runtime"][self.runtime]
        self.environment_variables = context.environment_variables
        self.perf_profile_flag = context.arguments["perf_profile"]
        self.profiling_level_flag = context.arguments["profiling_level"]
        self.debug_flag = context.arguments["debug_flag"]
        self.userlogs = context.arguments['userlogs']

        self.engine_path = context.engine_path
        self.target_arch = context.architecture
        self.target_path = context.output_dir
        if self.target_arch == "aarch64-android":
            self.target_path = context.target_path["htp"]
        elif self.target_arch == "wos-remote":
            self.remote_username = context.remote_username
            self.target_path = context.target_path["wos"].format(username=self.remote_username)

    def get_execute_environment_variables(self):

        def fill(variable):
            return variable.format(target_path=self.target_path, target_arch=self.target_arch)

        return {(k, fill(v)) for k, v in self.environment_variables.items()}

    def build_execute_command(self, container, input_list, userlogs, perf_profile=None,
                              profiling_level=None, extra_runtime_args=None, debug_mode=False):
        # type: (str, str) -> str
        execute_command_list = [
            self.executable, self.container, container, self.input_list, input_list,
            self.runtime_flag
        ]

        if perf_profile:
            execute_command_list.extend([self.perf_profile_flag, perf_profile])

        if profiling_level:
            execute_command_list.extend([self.profiling_level_flag, profiling_level])

        if extra_runtime_args:
            execute_command_list.extend([extra_runtime_args])

        if userlogs:
            execute_command_list.extend([self.userlogs, userlogs])

        if debug_mode:
            execute_command_list.append(self.debug_flag)

        execute_command_str = ' '.join(execute_command_list)
        return execute_command_str
