from setuptools import setup


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
    packages=["mavebay"],
    install_requires=[],
    include_package_data=True,
)
