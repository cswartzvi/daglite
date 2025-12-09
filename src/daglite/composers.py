"""
Deprecated module - composers functionality has been moved.

The split() function has been removed. Use the .split() method on TaskFuture instead:

    # Old:
    from daglite.composers import split
    a, b = split(future)

    # New:
    a, b = future.split()

The when() and loop() composers have been removed as they don't provide
sufficient value over implementing conditional logic within tasks.
"""

# This module is kept for backward compatibility but is now empty.
# Future versions may remove this module entirely.
