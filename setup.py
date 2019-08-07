from setuptools import setup

setup(name='rl_lib',
      version='0.1',
      description='Reinforcement Learning toolkit',
      url='https://github.com/jimkon/rl_lib',
      author='jimkon',
      author_email='dkontzedakis@gmail.com',
      license='MIT',
      install_requires=['numpy', 'tensorflow', 'matplotlib', 'gym', 'pandas'],
      packages=['rl_lib'],
      zip_safe=False)