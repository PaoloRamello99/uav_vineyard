from setuptools import setup
import os
from glob import glob

package_name = 'automatic_uav_mppi'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', 'automatic_uav_mppi', 'launch'),
        glob('launch/*.launch.py')),
    ],
    install_requires=[
        'setuptools',
        'uav_control_py',
    ],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='s319891@studenti.polito.it',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'offboard_mppi = automatic_uav_mppi.offboard_mppi:main',
        ],
    },
)
