"""Setup script for FLAME AI Challenge repository."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="flame-ai-challenge",
    version="1.0.0",
    author="AiREX lab",
    description="Machine Learning repository for Physics-Informed Super Resolution in the FLAME AI Challenge.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "flame-train=scripts.train:main",
            "flame-evaluate=scripts.evaluate:main",
            "flame-visualize=scripts.visualize:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# from setuptools import setup, find_packages

# setup(
#     name='fluid-sr',
#     version='0.1.0',
#     author='Diya',
#     description='A short description of your package.',
#     long_description=open('README.md').read(),
#     long_description_content_type='text/markdown',
#     packages=find_packages(),
#     install_requires=[
#         'requests',
#         'numpy>=1.20',
#     ],
#     classifiers=[
#         'Programming Language :: Python :: 3',
#         'License :: OSI Approved :: MIT License',
#         'Operating System :: OS Independent',
#     ],
#     entry_points={
#         'console_scripts': [
#             'my_script=my_package.cli:main',
#         ],
#     },
# )