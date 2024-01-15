from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_dependencies(filepath:str)->List[str]: 
    with open(filepath, "r") as f:
        requirements = f.readlines()
        requirements = [x.replace("\n","") for x in requirements]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name="TELCO_churn_classification",
    author="Dipan Banik",
    author_email="dipanthedataguy@gmail.com",
    version="0.0.1", # data.model.parameters
    packages=find_packages(),
    install_requires=get_dependencies("requirements.txt")
)