from setuptools import setup, find_packages

setup(
    name='fluid-sr',
    version='0.1.0',
    author='Diya',
    description='A short description of your package.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'requests',
        'numpy>=1.20',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'my_script=my_package.cli:main',
        ],
    },
)