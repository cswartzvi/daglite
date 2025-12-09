"""Pillow (PIL) serialization and hashing handlers."""

import hashlib

from PIL import Image

from daglite.serialization import default_registry


def hash_pil_image(img: Image.Image) -> str:
    """Fast hash for PIL Images using thumbnail.

    Strategy:
    - Hash size and mode (metadata)
    - Downsample to 32x32 for content hash

    This is much faster than hashing full resolution and catches
    all visible changes.

    Args:
        img: PIL Image

    Returns:
        SHA256 hex digest
    """
    h = hashlib.sha256()

    # Hash metadata
    h.update(str(img.size).encode())
    h.update(str(img.mode).encode())

    # Hash downsampled content
    thumb = img.resize((32, 32), Image.Resampling.LANCZOS)
    h.update(thumb.tobytes())

    return h.hexdigest()


def register_pillow_handlers():
    """Register PIL/Pillow image handlers with the default registry.

    This registers:
    - Hash strategy for PIL.Image.Image (thumbnail-based)

    Example:
        >>> from daglite_serialization.pillow import register_pillow_handlers
        >>> register_pillow_handlers()
    """
    # Register hash strategy
    default_registry.register_hash_strategy(
        Image.Image, hash_pil_image, "Thumbnail-based hash for PIL Images"
    )

    # Could also register serialization formats here if needed
    # For example: PNG, JPEG, WebP, etc.
