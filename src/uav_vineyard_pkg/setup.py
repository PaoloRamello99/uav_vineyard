from setuptools import setup, find_packages

package_name = 'uav_vineyard_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(include=[package_name]),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='s319891@studenti.polito.it',
    description='UAV vineyard package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'navigation = uav_vineyard_pkg.navigation:main',
            'monitor = uav_vineyard_pkg.monitor:main',
            'refill = uav_vineyard_pkg.refill:main',
            'sprayer = uav_vineyard_pkg.sprayer:main',
            'map = uav_vineyard_pkg.map:main',
            'mission_control = uav_vineyard_pkg.mission_control:main',
            'dashboard = uav_vineyard_pkg.dashboard:main',
        ],
    },
)
