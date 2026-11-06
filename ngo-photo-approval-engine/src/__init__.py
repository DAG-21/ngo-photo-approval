"""NGO Photo Approval Engine package."""

# iPhone photos are often HEIC files saved with a .jpg extension; register the HEIF decoder once so
# every PIL.Image.open in the package (ingestion, retrieval, Gemini client) can read them.
try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass
