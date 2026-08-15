from setuptools import setup

package_name = 'run_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rtree',
    maintainer_email='rtree@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'oyb=run_ai.only_yolo_ubuntu:main',
        'pdc=run_ai.patch_data_collector:main',
        'padcu=run_ai.Product_anomaly_detection_code_ubuntu:main',
        'ydc=run_ai.yolo_data_collector:main',
        ],
    },
)
