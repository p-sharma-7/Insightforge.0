from setuptools import find_packages, setup
from typing import List

Hyphen_E_dot='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function reads the requirements file and returns the list of requirements
    '''
    requirements=[]
    with open (file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements= [req.replace("\n","")for req in requirements]

        if Hyphen_E_dot in requirements:
            requirements.remove(Hyphen_E_dot)
    return requirements
    
setup(
name='Insightforge',
version='0.0.1',
author='Pushkar Sharma',
author_email= 'pushkarrokhel@gmail.com',
packages=find_packages(),
requirements = get_requirements('requirements.txt')
)