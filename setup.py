from pypescript.libutils import setup

setup(name='cosmopipe',
      base_dir='cosmopipe',
      author='Arnaud de Mattia et al.',
      maintainer='Arnaud de Mattia',
      url='http://github.com/adematti/cosmopipe',
      description='Cosmological library for pypescript',
      pype_module_names='install_modules.txt',
      license='GPLv3',
      install_requires=['matplotlib','scipy','tabulate'])
