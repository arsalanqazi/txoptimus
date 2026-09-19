from setuptools import setup, find_packages
from setuptools.command.install import install
import sys


class PostInstallCommand(install):
    """Post-installation: display requirements banner and environment warning."""

    def run(self):
        install.run(self)
        # Environment warning
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            env_type = "virtual environment"
        else:
            env_type = "system Python"

        print("\n" + "=" * 60)
        print(" TxOptimus v0.1.2 installed successfully!")
        print("=" * 60)

        if env_type == "system Python":
            print("\n ⚠ WARNING: You are installing into your SYSTEM Python.")
            print("   It is strongly recommended to install TxOptimus in a")
            print("   separate virtual or conda environment to avoid")
            print("   overwriting existing dependencies.")
            print("\n   Create a dedicated environment:")
            print("     conda create -n txoptimus python=3.8")
            print("     conda activate txoptimus")
            print("     pip install txoptimus")

        # Check if DGL is installed
        try:
            import dgl
            print(f"\n   ✓ DGL detected: {dgl.__version__}")
        except ImportError:
            print("\n ⚠ DGL is NOT installed. TxOptimus requires DGL 2.4.0.")
            print("   DGL must be installed separately (not available on PyPI):")
            print("")
            print("   # For CUDA 12.1:")
            print("   pip install https://data.dgl.ai/wheels/torch-2.4/cu121/dgl-2.4.0%2Bcu121-cp38-cp38-manylinux1_x86_64.whl")
            print("")
            print("   # For CPU only:")
            print("   pip install https://data.dgl.ai/wheels/torch-2.4/cpu/dgl-2.4.0-cp38-cp38-manylinux1_x86_64.whl")
            print("")
            print("   See https://www.dgl.ai/pages/start.html for other platforms.")

        print(f"\n   Detected environment: {env_type}")
        print(f"   Python: {sys.version}")

        print("""
 REQUIREMENTS:
   • RAM:     ≥32 GB (OptimusKG loads a 21.8M-edge graph)
   • Storage: ≥8 GB free disk space for data files
   • Python:  3.8+ (tested on 3.8.20)
   • GPU:     Optional (CPU inference supported)

 QUICK START:
   1. Download data:  txoptimus --setup
   2. Run inference:  txoptimus --engine optimus --diseases "oral cavity cancer"
   3. Benchmark:      txoptimus --engine benchmark --diseases "oral cavity cancer"

 For more details:  txoptimus --help
""")
        print("=" * 60 + "\n")


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='txoptimus',
    version='0.1.2',
    description='Zero-shot drug repurposing via TxGNN on PrimeKG and OptimusKG',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='arsalanriaz',
    url='https://github.com/arsalanqazi/txoptimus',
    python_requires='>=3.8',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch==2.4.0',
        # NOTE: dgl==2.4.0+cu121 must be installed separately (not on PyPI)
        # See post-install banner or README for instructions.
        'torch-geometric==2.6.1',
        'numpy==1.24.4',
        'pandas==2.0.3',
        'scikit-learn==1.3.2',
        'scipy==1.10.1',
        'matplotlib==3.7.5',
        'tqdm==4.70.0',
        'requests==2.32.4',
        'transformers',
        'networkx==3.1',
        'pydantic==2.10.6',
        'PyYAML==6.0.3',
        'openpyxl==3.1.5',
        'xlsxwriter==3.2.9',
        'goatools==1.6.5',
        'statsmodels==0.14.1',
        'rich==14.3.4',
        'pillow==10.4.0',
    ],
    entry_points={
        'console_scripts': [
            'txoptimus=txoptimus.txoptimus:main',
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    keywords='drug-repurposing zero-shot graph-neural-network knowledge-graph biomedical txgnn primekg optimuskg',
)
