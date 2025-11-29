"""
Declares pluggy markers daglite's hook specs and implementations.

For more information, please see: https://pluggy.readthedocs.io/en/stable/#marking-hooks.
"""

import pluggy

HOOK_NAMESPACE = "daglite"

hook_spec = pluggy.HookspecMarker(HOOK_NAMESPACE)
hook_impl = pluggy.HookimplMarker(HOOK_NAMESPACE)
