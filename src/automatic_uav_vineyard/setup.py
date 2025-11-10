from setuptools import setup
import os
from glob import glob

package_name = 'automatic_uav_vineyard'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='s319891@studenti.polito.it',
    description='Controllo offboard automatico per missione nel vigneto',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vineyard_offboard = automatic_uav_vineyard.vineyard_offboard:main',
        ],
    },
)
