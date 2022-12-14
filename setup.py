import os

import pkg_resources
from setuptools import find_packages, setup

setup(
    name="lora_diffusion",
    py_modules=["lora_diffusion"],
    version="0.0.4",
    description="Low Rank Adaptation for Diffusion Models. Works with Stable Diffusion out-of-the-box.",
    author="Simo Ryu",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "lora_add = lora_diffusion.cli_lora_add:main",
        ],
    },
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)
