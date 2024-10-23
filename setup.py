import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nmf_audio_benchmark",
    version="0.1.1",
    author="Marmoret Axel",
    author_email="axel.marmoret@imt-atlantique.fr",
    description="Benchmark for NMF methods on audio applications.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ax-le/nmf_audio_benchmark",    
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.7"
    ],
    license='BSD',
    install_requires=[
        'base_audio>=0.1.0',
        'doce>=0.1',
        'librosa >= 0.10.2',
        'matplotlib',
        'mir_eval',
        'mirdata',
        'nn_fac>= 0.3.2',
        'numpy',
        'scikit_learn',
        'tqdm',
    ],
)
