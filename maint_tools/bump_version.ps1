$PYPROJECT_FILE = "pyproject.toml"
$README_FILE = "README.rst"
$DOCS_VERSION_FILE = "docs/source/_static/versions.json"
$LATEST_CHANGES_FILE = "docs/source/changes/latest.rst"
$WHATS_NEW_FILE = "docs/source/changes/whats_new.rst"

if (-not (Test-Path $PYPROJECT_FILE)) {
    Write-Host "Error: pyproject.toml file not found!"
    exit 1
}

if (-not (Test-Path $README_FILE)) {
    Write-Host "Error: README.rst file not found!"
    exit 1
}

if (-not (Test-Path $DOCS_VERSION_FILE)) {
    Write-Host "Error: versions.json file not found!"
    exit 1
}

if (-not (Test-Path $LATEST_CHANGES_FILE)) {
    Write-Host "Error: latest.rst file not found!"
    exit 1
}

if (-not (Test-Path $WHATS_NEW_FILE)) {
    Write-Host "Error: whats_new.rst file not found!"
    exit 1
}

# Extract current version from pyproject.toml
$CURRENT_VERSION = Select-String -Path $PYPROJECT_FILE -Pattern '(?<=^version = ")[0-9]+\.[0-9]+\.[0-9]+(\.dev[0-9]*)?(?=")' | ForEach-Object { $_.Matches.Groups[0].Value }

# Remove .dev suffix from version
$NEW_VERSION = $CURRENT_VERSION -replace '\.dev', ''

# Update pyproject.toml with the new version
(Get-Content $PYPROJECT_FILE) -replace "version = `"$CURRENT_VERSION`"", "version = `"$NEW_VERSION`"" | Set-Content $PYPROJECT_FILE

# Update the version badge in README.md with the new version
(Get-Content $README_FILE) -replace "\/badge\/version-[0-9]+\.[0-9]+\.[0-9]+-orange", "/badge/version-$NEW_VERSION-orange" | Set-Content $README_FILE

# Update the versions.json file with a link to the new version docs
$versionsContent = Get-Content $DOCS_VERSION_FILE
$newVersionEntry = "    {`n        `"name`": `"$NEW_VERSION`",`n        `"version`": `"$NEW_VERSION`",`n        `"url`": `"http://10.113.113.118:8002/version/$NEW_VERSION/`"`n    },"
$versionsContent = $versionsContent[0..5] + $newVersionEntry + $versionsContent[6..($versionsContent.Length - 1)]
$versionsContent | Set-Content $DOCS_VERSION_FILE

# Remove the .dev suffix from the version section in the latest.rst file and add a line
# with the release date with format **Released Month Year**. Then rename this file to
# the new version number.
$EQUALS = "=" * $NEW_VERSION.Length
$latestChangesContent = Get-Content $LATEST_CHANGES_FILE
$latestChangesContent = $latestChangesContent -replace "$CURRENT_VERSION[0-9]*", "$NEW_VERSION"
$releaseDate = "`n**Released $(Get-Date -Format 'MMMM yyyy')**"
$latestChangesContent[3] = $EQUALS
$latestChangesContent = $latestChangesContent[0..3] + $releaseDate + $latestChangesContent[4..($latestChangesContent.Length - 1)]
$latestChangesContent | Set-Content $LATEST_CHANGES_FILE

# Rename the latest.rst file to the new version number
$LATEST_CHANGES_FILE = Resolve-Path $LATEST_CHANGES_FILE
$NEW_LATEST_CHANGES_FILE = Join-Path -Path (Get-Item $LATEST_CHANGES_FILE).DirectoryName -ChildPath  "$NEW_VERSION.rst"
Rename-Item -Path $LATEST_CHANGES_FILE -NewName $NEW_LATEST_CHANGES_FILE

# Update the whats_new.rst file with the new version
(Get-Content $WHATS_NEW_FILE) -replace "latest", "$NEW_VERSION" | Set-Content $WHATS_NEW_FILE

Write-Host "Version updated to $NEW_VERSION"
