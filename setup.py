from setuptools import setup

setup(
    name='pyArtacs',
    version='0.0.1',
    description='Toolbox for removal of periodic artifacts.',
    long_description='A Python toolbox for removal of artifact caused by transcranial periodic, especially alternating current stimulation (tACS).',
    author='Robert Guggenberger',
    author_email='robert.guggenberger@uni-tuebingen.de',
    url='https://github.com/agricolab/pyArtacs',
    download_url='https://github.com/agricolab/pyArtacs.git',
    license='MIT',
    packages=['artacs'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Brain Stimulation',             
        'Topic :: Scientific/Engineering :: Signal Processing',
        ]
)
