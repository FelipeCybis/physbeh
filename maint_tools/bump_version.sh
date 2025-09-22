#!/bin/bash

PYPROJECT_FILE="pyproject.toml"
README_FILE="README.rst"
DOCS_VERSION_FILE="docs/source/_static/versions.json"
LATEST_CHANGES_FILE="docs/source/changes/latest.rst"
WHATS_NEW_FILE="docs/source/changes/whats_new.rst"

if [ ! -f "$PYPROJECT_FILE" ]; then
    echo "Error: pyproject.toml file not found!"
    exit 1
fi

if [ ! -f "$README_FILE" ]; then
    echo "Error: README.rst file not found!"
    exit 1
fi

if [ ! -f "$DOCS_VERSION_FILE" ]; then
    echo "Error: versions.json file not found!"
    exit 1
fi

if [ ! -f "$LATEST_CHANGES_FILE" ]; then
    echo "Error: latest.rst file not found!"
    exit 1
fi

if [ ! -f "$WHATS_NEW_FILE" ]; then
    echo "Error: whats_new.rst file not found!"
    exit 1
fi

# Extract current version from pyproject.toml
CURRENT_VERSION=$(grep -Po '(?<=^version = ")[0-9]+\.[0-9]+\.[0-9]+(\.dev[0-9]*)?(?=")' "$PYPROJECT_FILE")

# Remove .dev suffix from version
NEW_VERSION=$(echo "$CURRENT_VERSION" | sed 's/\.dev//')

# Update pyproject.toml with the new version
sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" "$PYPROJECT_FILE"

# Update the version badge in README.md with the new version
sed -i "s/\/badge\/version-[0-9]\+\.[0-9]\+\.[0-9]\+-orange/\/badge\/version-$NEW_VERSION-orange/" "$README_FILE"

# Update the versions.json file with a link to the new version docs
sed -i "7i\    {\n        \"name\": \"$NEW_VERSION\",\n        \"version\": \"$NEW_VERSION\",\n        \"url\": \"http://10.113.113.118:8000/version/$NEW_VERSION/\"\n    }," "$DOCS_VERSION_FILE"

# Remove the .dev suffix from the version section in the latest.rst file and add a line
# with the release date with format **Released Month Year**. Then rename this file to
# the new version number.
EQUALS=$(printf '=%.0s' $(seq 1 ${#NEW_VERSION}))
sed -i "3s/$NEW_VERSION\(\.dev[0-9]*\)/$NEW_VERSION/" "$LATEST_CHANGES_FILE"
sed -i "4s/=*/$EQUALS\n\n**Released $(date +'%B %Y')**/" "$LATEST_CHANGES_FILE"
mv "$LATEST_CHANGES_FILE" "docs/source/changes/$NEW_VERSION.rst"

# Update the whats_new.rst file with the new version
sed -i "s/latest/$NEW_VERSION/" "$WHATS_NEW_FILE"

echo "Version updated to $NEW_VERSION"
