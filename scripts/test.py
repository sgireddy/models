# Validation script to verify environment and AI Edge imports on macOS
import sys, os
print('Python:', sys.version)

# Core libs
import torch
import transformers
print('torch:', torch.__version__)
print('transformers:', transformers.__version__)

# Try setting DYLD_LIBRARY_PATH to ai_edge_litert package directory before import
litert_dir = None
try:
    import importlib.util
    spec = importlib.util.find_spec('ai_edge_litert')
    if spec and spec.origin:
        litert_dir = os.path.dirname(spec.origin)
        os.environ['DYLD_LIBRARY_PATH'] = f"{litert_dir}:{os.environ.get('DYLD_LIBRARY_PATH','')}"
        print('Set DYLD_LIBRARY_PATH to:', os.environ['DYLD_LIBRARY_PATH'])
except Exception as e:
    print('Failed to locate ai_edge_litert before import:', repr(e))

# Try AI Edge Torch generative utilities without importing top-level package
try:
    from ai_edge_torch.generative.utilities import converter
    from ai_edge_torch.generative.utilities.export_config import ExportConfig
    print('ai_edge_torch.generative.utilities imports: OK')
except Exception as e:
    print('ai_edge_torch generative utilities import FAILED:', repr(e))
    if litert_dir and os.path.isdir(litert_dir):
        print('ai_edge_litert directory listing:', os.listdir(litert_dir))
        print('Has libpywrap_litert_common.dylib:', any('libpywrap_litert_common' in f for f in os.listdir(litert_dir)))

# Try ai_edge_litert interpreter (root cause of prior error)
try:
    from ai_edge_litert import interpreter as tfl_interpreter
    print('ai_edge_litert interpreter import: OK')
except Exception as e:
    print('ai_edge_litert interpreter import FAILED:', repr(e))
    if litert_dir and os.path.isdir(litert_dir):
        print('ai_edge_litert directory listing:', os.listdir(litert_dir))
        print('Has libpywrap_litert_common.dylib:', any('libpywrap_litert_common' in f for f in os.listdir(litert_dir)))
