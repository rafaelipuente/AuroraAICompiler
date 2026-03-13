import os
import lit.formats

config.name = "Aurora"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.aurora_obj_root, "test")

# Minimal standalone lit setup: prepend tool directories to PATH so RUN lines
# can invoke aurora-opt, FileCheck, and not directly.
path_entries = [config.aurora_tools_dir, config.llvm_tools_dir]
existing_path = config.environment.get("PATH", "")
config.environment["PATH"] = os.pathsep.join(path_entries + [existing_path])
