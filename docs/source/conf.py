# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import inspect
import operator
import os
import sys

import physbeh

sys.path.append(os.path.abspath("./_ext"))
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "physbeh"
copyright = "2023, Felipe Cybis Pereira"
author = "Felipe Cybis Pereira"

current_version = physbeh.__version__

# Latest release version
latest_release = (
    os.popen(
        "git describe --tags " + os.popen("git rev-list --tags --max-count=1").read()
    )
    .read()
    .strip()
)

# Cname of the project
cname = os.getenv("DOCUMENTATION_CNAME", "http://10.113.113.118:8002")
# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "numpydoc",
    "sphinx.ext.linkcode",
    "gh_substitutions",
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
    "Tracking": "physbeh.tracking.Tracking",
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
html_title = "PhysBeh"
html_short_title = "PhysBeh"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "no_italic.css",  # disable italic for span classes
    "theme_layout.css",  # control layout width
]


def get_version_match(semver):
    """Evaluate the version match for the multi-documentation."""
    if semver.endswith("dev0"):
        return "dev"
    return semver


html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/FelipeCybis/physbeh",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": True,
    "check_switcher": False,
    "switcher": {
        "json_url": f"{cname}/version/latest/_static/versions.json",
        "version_match": get_version_match(current_version),
    },
    "navbar_end": ["version_switcher", "theme-switcher", "navbar-icon-links"],
}
html_context = {
    "github_url": "https://github.com",
    "github_user": "FelipeCybis",
    "github_repo": "physbeh",
    "github_version": "main",
    "doc_path": "docs/source",
}
# Add banner in case version is not stable
if "dev" in current_version:
    html_theme_options["announcement"] = (
        "<p>This is the development documentation "
        f"of PhysBeh ({current_version}) "
        '<a class="sd-sphinx-override sd-badge sd-text-wrap '
        'sd-btn-outline-dark reference external" '
        f'href="{cname}/version/{latest_release}">'
        f"<span>Switch to stable version ({latest_release})</span></a></p>"
    )


def touch_example_backreferences(app, what, name, obj, options, lines):
    # generate empty examples files, so that we don't get
    # inclusion errors if there are no examples for a class / module
    examples_path = os.path.join(app.srcdir, "api", "generated", f"{name}.examples")
    if not os.path.exists(examples_path):
        # touch file
        open(examples_path, "w").close()


def setup(app):
    app.connect("autodoc-process-docstring", touch_example_backreferences)


# The following is used by sphinx.ext.linkcode to provide links to github
def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None

    revision = "main"
    package = "physbeh"
    url_fmt = (
        "https://github.com/FelipeCybis/physbeh/blob/{revision}/"
        "src/{package}/{path}#L{lineno}"
    )

    class_name = info["fullname"].split(".")[0]
    module = __import__(info["module"], fromlist=[class_name])
    obj = operator.attrgetter(info["fullname"])(module)
    # Unwrap the object to get the correct source
    # file in case that is wrapped by a decorator
    obj = inspect.unwrap(obj)

    try:
        # get filepath from object
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        return

    # Don't include filenames from outside this package's tree
    if os.path.dirname(__import__(package).__file__) not in fn:
        return

    # get filepath relative to package root (will be the same in github)
    fn = os.path.relpath(fn, start=os.path.dirname(__import__(package).__file__))
    try:
        # get permalink of object, if possible
        lineno = inspect.getsourcelines(obj)[1]
    except Exception:
        lineno = ""
    return url_fmt.format(revision=revision, package=package, path=fn, lineno=lineno)
