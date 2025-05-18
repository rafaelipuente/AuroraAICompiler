from setuptools import setup, find_packages

setup(
    name="aurora",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "onnx>=1.12.0",
        "torch>=2.0.0",
        "tqdm>=4.64.0",
        "pytest>=7.0.0"
    ],
    author="Aurora AI Team",
    author_email="info@auroraai.example.com",
    description="Python bindings and utilities for AuroraAICompiler",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/AuroraAICompiler",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
