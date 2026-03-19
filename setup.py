from setuptools import setup, find_packages

setup(
    name="graph-diffusion-edge-serving",
    version="0.1.0",
    description="Efficient GNN inference via precomputed PPR diffusion matrices",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scipy>=1.10",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
