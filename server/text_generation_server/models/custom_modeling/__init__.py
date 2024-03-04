import os

DISABLE_CUSTOM_KERNELS = (
    os.environ.get("DISABLE_CUSTOM_KERNELS", "False").lower() == "true"
)
