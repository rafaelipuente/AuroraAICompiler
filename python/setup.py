from setuptools import setup, find_packages

setup(
    name="aurora",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "onnx>=1.12.0",
    ],
    extras_require={
        "test": ["pytest>=7.0.0"],
    },
    author="Rafael Puente",
    author_email="",
    description="Python utilities for AuroraAICompiler (ONNX import, IR generation)",
    url="https://github.com/rafaelipuente/AuroraAICompiler",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
