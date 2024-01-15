from setuptools import find_packages, setup

def get_dependencies(filepath):
    with open(filepath, "r") as f:
        requirements = f.readlines()
        requirements = [x.replace("/n","") for x in requirements]
    return requirements


setup(
    name="TELCO_churn_classification",
    author="Dipan Banik",
    version="0.0.1", # data.model.parameters
    packages=find_packages(),
    install_requires=get_dependencies("requirements.txt")
)