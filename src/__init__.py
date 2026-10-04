"""
Source package namespace for Adaptive Edge-AI Navigation System.
Provides a clean modular interface matching the target workspace architecture,
while maintaining full compatibility with the canonical adaptive_navigation package.
"""

import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import adaptive_navigation

__all__ = ["adaptive_navigation"]
