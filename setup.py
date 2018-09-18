from distutils.core import setup


setup(
    name='pyArtacs',
    version='0.0.1',
    description='Toolbox for removal of transcranial periodic current stimulation artifacts.',
    long_description='A Python toolbox for periodic tCS artifact removal.',
    author='Robert Guggenberger',
    author_email='robert.guggenberger@uni-tuebingen.de',
    url='https://github.com/agricolab/pyArtacs'
    download_url='https://github.com/agricolab/pyArtacs.git',
    license='MIT',
    packages=['artacs'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Libraries',
        ]
)
