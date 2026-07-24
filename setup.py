from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="raredis-graph-multiagent",
    version="1.0.0",
    author="Quan Minh Nguyen, Kai Wang",
    author_email="nguyenqm@chop.edu",
    description=(
        "A locally deployable multi-agent system for rare disease "
        "prioritization using a curated knowledge graph and open-weight "
        "language models"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WGLab/RareDisGraph-MultiAgentLLM",
    project_urls={
        "Bug Tracker": "https://github.com/WGLab/RareDisGraph-MultiAgentLLM/issues",
        "Knowledge Graph": "https://github.com/WGLab/RareDisGraph-Extraction",
        "Lab": "https://github.com/WGLab",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    keywords=[
        "rare disease",
        "phenotype prioritization",
        "knowledge graph",
        "language model",
        "clinical NLP",
        "HPO",
        "MONDO",
        "diagnostic support",
    ],
    entry_points={
        "console_scripts": [
            "raredis-run=raregraph.cli:run_pipeline",
            "raredis-batch=raregraph.cli:run_batch",
            "raredis-evaluate=raregraph.cli:evaluate",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
