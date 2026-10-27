from pathlib import Path
from setuptools import find_packages, setup

# Read requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

# Read long description from README.md
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="smart_lemon_harvester",
    version="0.1.0",
    author="Yogesh",
    description="AI-powered lemon detection, ripeness classification, and quality grading for smart harvesting systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yogesh/smart-lemon-harvester",
    packages=find_packages(exclude=["tests*", "runs*", "data*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    include_package_data=True,
)
