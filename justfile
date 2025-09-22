set windows-shell := ["pwsh.exe", "-c"]

# Print the help message.
@help:
    echo "Usage: just [RECIPE]\n"
    just --list

# Run all tests.
test *args:
    uv run --all-extras pytest {{args}}

docs:
    uv run --extra docs sphinx-build -j auto docs/source/ docs/build/

clean-docs:
    rm -rf docs/build/
    rm -rf docs/source/api/generated/
    rm -rf docs/source/auto_examples/

release_script := if os() == "windows" { "maint_tools/bump_version.ps1" } else { "maint_tools/bump_version.sh" }

# Bump the version and prepare a new release.
release:
   {{ release_script }} 
