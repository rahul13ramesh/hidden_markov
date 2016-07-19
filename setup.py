from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='hidden_markov',
      version='0.3',
      description='Implementation of Hidden markov model in discrete domain.',
      url='http://github.com/Red-devilz/hidden_markov',
      author='Rahul',
      author_email='',
      license='MIT',
      packages=['hidden_markov'],
      install_requires=[
          'numpy',
      ],

      classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',


        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

      test_suite='nose.collector',
      tests_require=['nose'],

      long_description=readme(),

      keywords='hmm hidden markov model',

      zip_safe=False)
