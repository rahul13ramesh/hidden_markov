from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='hidden_markov',
      version='0.1',
      description='Implementation of Hidden markov model in discrete domain.',
      url='http://github.com/Red-devilz/hidden_markov',
      author='Rahul',
      author_email='',
      license='MIT',
      packages=['hidden_markov'],
      install_requires=[
          'numpy',
      ],

      long_description=readme(),

      keywords='hmm markov model',

      zip_safe=False)
