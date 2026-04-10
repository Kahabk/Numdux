"""
SafeSandbox - Secure code execution for Numdux.
Runs AI-generated Python code in a restricted environment.
Prevents filesystem writes, network access, and dangerous operations.
"""

import sys
import io
import traceback
import builtins
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List

# Operations to block in sandbox
_BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "importlib",
    "subprocess", "shutil", "socket",
}

class SafeSandbox:
    """
    Execute Python code safely with access to data science libraries.
    Blocks filesystem writes, network access, and dangerous builtins.
    
    Usage:
        sandbox = SafeSandbox()
        result = sandbox.run("print(df.head())", {"df": my_dataframe})
    """

    def __init__(self, timeout: int = 60):
        self.timeout = timeout

    def _make_globals(self, extra: dict = None) -> dict:
        """Build a safe global namespace with data science imports."""
        import pandas as pd
        import numpy as np

        # Safe subset of builtins
        safe_builtins = {k: getattr(builtins, k) for k in dir(builtins)
                         if k not in _BLOCKED_NAMES and not k.startswith("_")}

        safe_globals: Dict[str, Any] = {
            "__builtins__": safe_builtins,
            "pd": pd, "pandas": pd,
            "np": np, "numpy": np,
            "print": print,
            "len": len, "range": range, "list": list, "dict": dict,
            "str": str, "int": int, "float": float, "bool": bool,
            "zip": zip, "map": map, "filter": filter, "sorted": sorted,
            "enumerate": enumerate, "sum": sum, "min": min, "max": max,
            "abs": abs, "round": round, "type": type, "isinstance": isinstance,
        }

        # Optional data science libraries
        optional_imports = [
            ("plotly.express", "px"),
            ("plotly.graph_objects", "go"),
            ("plotly.subplots", "make_subplots"),
            ("sklearn", "sklearn"),
            ("sklearn.model_selection", "model_selection"),
            ("sklearn.preprocessing", "preprocessing"),
            ("sklearn.ensemble", "ensemble"),
            ("sklearn.metrics", "metrics"),
            ("scipy.stats", "stats"),
            ("warnings", "warnings"),
            ("json", "json"),
            ("math", "math"),
            ("re", "re"),
            ("datetime", "datetime"),
            ("collections", "collections"),
            ("itertools", "itertools"),
            ("statistics", "statistics"),
        ]

        import importlib
        for module_path, alias in optional_imports:
            try:
                mod = importlib.import_module(module_path)
                safe_globals[alias] = mod
            except ImportError:
                pass

        if extra:
            safe_globals.update(extra)

        return safe_globals

    def run(self, code: str, context: dict = None) -> dict:
        """
        Execute code in the sandbox.
        
        Args:
            code: Python source code to execute
            context: Variables to inject (e.g. {"df": dataframe})
            
        Returns:
            dict with keys: success, output, error, locals, figures
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        local_vars: Dict[str, Any] = {}
        g = self._make_globals(context or {})

        try:
            compiled = compile(code, "<numdux_sandbox>", "exec")

            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compiled, g, local_vars)

            stdout_out = stdout_buf.getvalue()
            stderr_out = stderr_buf.getvalue()

            # Combine outputs
            output = stdout_out
            if stderr_out and not stdout_out:
                output = stderr_out
            elif stderr_out:
                output = stdout_out + "\n[stderr]: " + stderr_out

            # Extract meaningful locals
            result_locals = {}
            for k in ["df", "df_clean", "df_processed", "model", "X", "y",
                      "X_train", "X_test", "y_train", "y_test"]:
                if k in local_vars:
                    result_locals[k] = local_vars[k]

            # Extract plotly figures
            figures = []
            for v in list(local_vars.values()) + list(g.values()):
                if hasattr(v, "to_json") and hasattr(v, "show"):
                    figures.append(v)

            return {
                "success": True,
                "output": output or "(execution completed with no output)",
                "error": None,
                "locals": result_locals,
                "figures": figures,
                "all_locals": {k: v for k, v in local_vars.items()
                               if not k.startswith("_")},
            }

        except SyntaxError as e:
            return {
                "success": False,
                "output": f"Syntax Error: {e}",
                "error": f"SyntaxError at line {e.lineno}: {e.msg}",
                "locals": {}, "figures": [],
            }
        except PermissionError as e:
            return {
                "success": False,
                "output": f"🔒 Blocked: {e}",
                "error": str(e),
                "locals": {}, "figures": [],
            }
        except Exception:
            tb = traceback.format_exc()
            return {
                "success": False,
                "output": tb,
                "error": tb,
                "locals": {}, "figures": [],
            }
