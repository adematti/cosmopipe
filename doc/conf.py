# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

lib_dir = os.path.abspath(os.path.join('..','cosmopipe'))
sys.path.insert(0,lib_dir)

from _version import __version__, __docker_image__

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme'
]

#import lib # hack to avoid issues with meta classes
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# -- Project information -----------------------------------------------------

project = 'cosmopipe'
copyright = '2021, Arnaud de Mattia'
author = 'Arnaud de Mattia'

# The full version, including alpha/beta/rc tags
release = __version__

html_theme = 'sphinx_rtd_theme'

autodoc_mock_imports = ['cosmoprimo']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', '**.ipynb_checkpoints']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
#html_css_files = ['css/custom.css']
html_style = 'css/custom.css'

git_repo = 'https://github.com/adematti/cosmopipe.git'
git_root = 'https://github.com/adematti/cosmopipe/blob/main/'
docker_root = 'https://hub.docker.com/r/adematti/cosmopipe/'

extlinks = {'root': (git_root + '%s',''),
            'dockerroot': (docker_root + '%s','')}

intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy/', None)
}

# thanks to: https://github.com/sphinx-doc/sphinx/issues/4054#issuecomment-329097229
def _replace(app, docname, source):
    result = source[0]
    for key in app.config.ultimate_replacements:
        result = result.replace(key, app.config.ultimate_replacements[key])
    source[0] = result


ultimate_replacements = {
    '{dockerimage}': __docker_image__,
    '{gitrepo}': git_repo
}


def setup(app):
    try:
        from pypescript.libutils import write_pype_modules_rst_doc
        write_pype_modules_rst_doc(os.path.join('api','modules.rst'),header='.. _api-modules:\n\n',max_line_len=150,base_dir=lib_dir)
    except ImportError:
        pass
    app.add_config_value('ultimate_replacements', {}, True)
    app.connect('source-read',_replace)


autoclass_content = 'both'
