from setuptools import setup

setup(name='rl_lib',
      version='0.1',
      description='Reinforcement Learning toolkit',
      url='https://github.com/jimkon/rl_lib',
      author='jimkon',
      author_email='dkontzedakis@gmail.com',
      license='MIT',
      packages=['rl_lib', 'numpy', 'tensorflow', 'matplotlib'],
      zip_safe=False)