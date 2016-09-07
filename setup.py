from setuptools import setup

setup(
	include_package_data=True,
	name='linear_bayes',
      version='0.1',
      description='Bayesian linear regression',
      url='https://github.com/anguswilliams91/LinearBayes',
      author='Angus Williams',
      author_email='anguswilliams91@gmail.com',
      license='MIT',
      packages=['linear_bayes'],
      package_dir={'linear_bayes': 'linear_bayes'},
      package_data={'linear_bayes': 'data/mock_data.npy'},
      install_requires = ['emcee']
	)