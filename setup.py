import os
from glob import glob
from setuptools import setup, find_packages

package_name = 'yolact_ros2'

setup(
    name=package_name,
    version='1.1.0',
    # Packages to export
    packages=[package_name],
    # Files we want to install, specifically launch files
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        # Include our package.xml file
        (os.path.join('share', package_name), ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    # This is important as well
    install_requires=['setuptools'],
    zip_safe=True,
    author='Fernando Gonzalez',
    author_email='fergonzaramos@yahoo.es',
    maintainer='Fernando Gonzalez',
    maintainer_email='fergonzaramos@yahoo.es',
    keywords=['ROS2'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: BSD',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='The yolact_ros2 package',
    license='BSD',
    # Like the CMakeLists add_executable macro, you can add your python
    # scripts here.
    entry_points={
        'console_scripts': [
            'yolact_ros2_node = yolact_ros2.yolact_ros2:main',
        ],
    },
)
