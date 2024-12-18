# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

import tracking_physmed

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "tracking_physmed"
copyright = "2023, Felipe Cybis Pereira"
author = "Felipe Cybis Pereira"

current_version = tracking_physmed.__version__
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

nitpicky = True
# Autosummary configuration
autosummary_generate = True
autosummary_context = {
    # Methods that should be skipped when generating the docs.
    # Class docstring should be under the instance itself and not in __init__.
    "skipmethods": ["__init__", "initialize"],
    "skipclassesmembers": [],
}

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# Once we have type hints for everything we can decide what to do with them in the
# documentation, for now numpydoc does not handle them very well so we set them not to
# appear in the description of the function
autodoc_typehints = "none"

# -- numpydoc configuration ------------------------------------------------------------

# Get rid of spurious warnings due to some interaction between autosummary and numpydoc.
# See https://github.com/phn/pytpm/issues/3#issuecomment-12133978 for more details
numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "Tracking": "tracking_physmed.tracking.Tracking",
    "path-like": ":term:`path-like <python:path-like object>`",
    "path_like": ":term:`path-like <python:path-like object>`",
    "function": "callable",
}
numpydoc_xref_ignore = {"optional", "or", "of"}
numpydoc_validate = True
# error_ignores = {
#     "GL01",  # Docstring should start in the line immediately after the quotes
#     "EX01",  # No examples section found
#     "ES01",  # No extended summary found
#     "SA01",  # See Also section not found
# }
# numpydoc_validation_checks = {"all"} | error_ignores
# numpydoc_validation_exclude = {
#     r"\.__init__$",
#     # Ignore anything that's private (e.g., starts with _)
#     r"\._.*$",
#     # Ignore methods inherited from sklearn.base.BaseEstimator
#     "FirstLevelModel.fit_transform",
#     "SecondLevelModel.fit_transform",
#     r"\.get_metadata_routing",
#     r"\.get_params",
#     r"\.set_params",
#     r"\.set_fit_request",
#     r"\.set_transform_request",
#     r"\.set_inverse_transform_request",
#     # Ignore methods inherited from sklearn.base.BaseEstimator
#     r"\.set_output",
#     r"AnimateScan\.to_html5_video",
#     r"AnimateScan\.to_jshtml",
#     r"AnimateScan\.new_saved_frame_seq",
#     r"AnimateScan\.new_frame_seq",
#     r"AnimateScan\.save",
#     r"Animation\.save",
# }

# -- Intersphinx configuration ---------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/devdocs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

templates_path = ["_templates"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = "TrackingPhysmed"
html_short_title = "TrackingPhysmed"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "no_italic.css",  # disable italic for span classes
    "theme_layout.css",  # control layout width
]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Iconeus/pythmed",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": True,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_context = {
    "github_url": "https://github.com",
    "github_user": "FelipeCybis",
    "github_repo": "tracking_physmed",
    "github_version": "main",
    "doc_path": "docs/source",
}
