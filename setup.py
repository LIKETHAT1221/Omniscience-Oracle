from setuptools import setup, find_packages

setup(
    name="omniscience-oracle",
    version="0.1.0",
    description="Advanced sports betting analysis system with technical indicators and AI-powered recommendations",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.28.1",
        "pandas==1.5.3",
        "numpy==1.24.3",
        "plotly==5.15.0",
    ],
    python_requires=">=3.7",
)
