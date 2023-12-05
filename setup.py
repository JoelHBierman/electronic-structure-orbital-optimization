import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as fp:
    install_requires = fp.read()

setuptools.setup(
    name="orbital-optimization",
    description="A collection of quantum algorithms to solve the electronic structure problem using orbital optimization.",
    long_description=long_description,
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires='>=3.8',
    version='0.2.0'
)
