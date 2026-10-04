"""
Convenience launcher for the Adaptive Navigation Pipeline.
Passes arguments through to main.py.
"""

import sys
import os
import subprocess

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
main_py = os.path.join(_repo_root, "adaptive_navigation", "main.py")

if __name__ == "__main__":
    cmd = [sys.executable, main_py] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
