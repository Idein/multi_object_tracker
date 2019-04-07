import os
from setuptools import setup, find_packages

exec(open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'multi_object_tracker', '_version.py')).read())

with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='multi-object-tracker',
    version=__version__,
    description='Python Library for Multiple Object Tracking',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Idein/multi_object_tracker',
    author='Idein Inc.',
    author_email='koichi@idein.jp',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='object tracking computer vision',
    packages=find_packages(),
    install_requires=['numpy', 'lapsolver']
)
