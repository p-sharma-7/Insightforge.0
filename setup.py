from setuptools import setup, find_packages

def get_requirements(filename):
    with open(filename, encoding='utf-8') as file_obj:
        requirements = file_obj.readlines()
    return [req.strip() for req in requirements]

setup(
    name="Insightforge",
    version="0.0.1",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)