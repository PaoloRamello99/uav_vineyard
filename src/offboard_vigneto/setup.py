from setuptools import setup

package_name = 'offboard_vigneto'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='s319891@studenti.polito.it',
    description='Controllo offboard per drone su filari di vigneto',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'offboard_vigneto_control = offboard_vigneto.offboard_vigneto_control:main',
        ],
    },
)
