from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'automatic_uav_mppi'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', 'automatic_uav_mppi', 'launch'),
        glob('launch/*.launch.py')),
        (os.path.join("share", package_name, "config"),
            glob("automatic_uav_mppi/config/*.yaml"),),
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
            'uav_offboard_mppi = automatic_uav_mppi.uav_offboard_mppi:main', # <--- ADD COMMA HERE
            'mppi_rate_node = automatic_uav_mppi.mppi_rate_node:main',
            'reference_node = automatic_uav_mppi.reference_node:main',
            'serpentine_trajectory = automatic_uav_mppi.serpentine_trajectory:main',
            'Enu2Ned = automatic_uav_mppi.coordinates_conversion.Enu2Ned:main',
            'Ned2Enu = automatic_uav_mppi.coordinates_conversion.Ned2Enu:main',
            'lemniscate_node = automatic_uav_mppi.lemniscate_node:main',
        ],
    },
)