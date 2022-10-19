from setuptools import setup, find_packages
PACKAGES = find_packages()

__version__ = '0.0.1'

if __name__ == '__main__':

    setup(
        name="contrastive-user-encoder",
        version=__version__,
        description="Training contrastive user encoders from Reddit posts",
        maintainer='Roberta Rocca',
        maintainer_email='rbrrcc@gmail.com',
        url='http://github.com/rbroc/personality_reddit',
        install_requires=['numpy>=1.19.5', 'pandas>=1.1.0', 
                          'tensorflow==2.4.1', 'markdown2',
                          'keras==2.4.0', 'seaborn>=0.11.0', 
                          'matplotlib', 'tf-models-official', 
                          'transformers==4.2.0',
                          'zstd', 'fasttext', 'pydot'], 
        packages=find_packages(exclude=['tests']),
        license='MIT',
        zip_safe=False,
        download_url=(f"https://github.com/rbroc/contrastive-user-encoder/archive/{__version__}.tar.gz"),
        python_requires='>=3.6',
    )
