from setuptools import find_packages, setup

package_name = 'automatic_uav_vineyard'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
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
            'vineyard_offboard = automatic_uav_vineyard.vineyard_offboard:main',
            'monitor = automatic_uav_vineyard.monitor:main',
            'refill = automatic_uav_vineyard.refill:main',
            'sprayer = automatic_uav_vineyard.sprayer:main',
            'map = automatic_uav_vineyard.map:main',
            'mission_control = automatic_uav_vineyard.mission_control:main',
            'dashboard = automatic_uav_vineyard.dashboard:main',
        ],
    },
)
