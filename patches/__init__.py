"""Patches to add TIGER-speech model support to pymms/pymss_core.

These patches are applied at worker startup time (via ``worker_protocol.py``)
so the pip-installed ``pymss`` and ``pymss_core`` packages gain TIGER support
*without* modifying any installed file.

Usage
-----
Called automatically when the worker starts.  No manual action required.

``from patches.apply_all import apply_all; apply_all()``
"""
