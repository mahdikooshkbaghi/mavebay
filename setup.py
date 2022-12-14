from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="mavebay",
    version="0.0.1",
    description="Bayesian inference for genotype-phenotype maps from multiplex assays of variant effect",
    long_description=readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/mahdikooshkbaghi/mavebay",
    author="Mahdi Kooshkbaghi",
    author_email="mahdi.kooshkbaghi@gmail.com",
    keywords="MAVE, bayesian inference",
    license="MIT",
    packages=find_packages(include=["mavebay", "mavebay.*"]),
    install_requires=[
        "absl-py==1.2.0",
        "attrs==22.1.0",
        "black==22.6.0",
        "click==8.1.3",
        "coverage==6.4.3",
        "etils==0.7.1",
        "flake8==5.0.4",
        "importlib-resources==5.9.0",
        "iniconfig==1.1.1",
        "isort==5.10.1",
        "jax==0.3.15",
        "jaxlib==0.3.15",
        "mccabe==0.7.0",
        "multipledispatch==0.6.0",
        "mypy-extensions==0.4.3",
        "numpy>=1.21.0",
        "numpyro==0.10.0",
        "opt-einsum==3.3.0",
        "packaging==21.3",
        "pandas>=1.3.0",
        "pathspec==0.9.0",
        "platformdirs==2.5.2",
        "pluggy==1.0.0",
        "py==1.11.0",
        "pycodestyle==2.9.1",
        "pyflakes==2.5.0",
        "pyparsing==3.0.9",
        "pytest==7.1.2",
        "pytest-cov==3.0.0",
        "python-dateutil==2.8.2",
        "pytz==2022.2.1",
        "scipy>=1.7.0",
        "six==1.16.0",
        "tomli==2.0.1",
        "tqdm==4.64.0",
        "typing_extensions==4.3.0",
        "zipp==3.8.1",
        "optax>=0.1.2",
        "scikit-learn>=1.1",
    ],
    include_package_data=True,
    extras_require={  # To install additional packages for examples  pip install . "mavebay[examples]"
        "examples": [  # Requirements only for examples not the base code.
            "arviz",
            "jupyter",
            "matplotlib",
            "seaborn",
            "logomaker",
        ],
    },
)
