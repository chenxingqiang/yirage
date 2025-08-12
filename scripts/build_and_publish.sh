#!/bin/bash

# Exit on error
set -e

# Script to build and publish the yica-yirage package
# This script ensures version consistency and builds the package for PyPI

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"  # Go up two levels to get to the project root

echo -e "${GREEN}=== YICA-Yirage Package Build and Publish Script ===${NC}"
echo "Project root: $PROJECT_ROOT"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    exit 1
fi

# Check if build and twine are installed
if ! pip3 list | grep -q "build"; then
    echo -e "${YELLOW}Installing build package...${NC}"
    pip3 install build
fi

if ! pip3 list | grep -q "twine"; then
    echo -e "${YELLOW}Installing twine package...${NC}"
    pip3 install twine
fi

# Ensure version consistency
echo -e "${GREEN}Checking version in pyproject.toml...${NC}"
PYPROJECT_VERSION=$(grep -m 1 'version = ' "$PROJECT_ROOT/pyproject.toml" | sed -E 's/version = "([^"]+)"/\1/')
echo "pyproject.toml version: $PYPROJECT_VERSION"

echo -e "${GREEN}Checking version in version.py...${NC}"
VERSION_PY_PATH="$PROJECT_ROOT/yirage/python/yirage/version.py"
if [ -f "$VERSION_PY_PATH" ]; then
    VERSION_PY_VERSION=$(grep -m 1 '__version__ = ' "$VERSION_PY_PATH" | sed -E 's/__version__ = "([^"]+)"/\1/')
    echo "version.py version: $VERSION_PY_VERSION"
else
    echo -e "${RED}Error: version.py not found at $VERSION_PY_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}Checking version in __init__.py...${NC}"
INIT_PY_PATH="$PROJECT_ROOT/yirage/python/yirage/__init__.py"
if [ -f "$INIT_PY_PATH" ]; then
    INIT_PY_VERSION=$(grep -m 1 '__version__ = ' "$INIT_PY_PATH" | sed -E 's/__version__ = "([^"]+)"/\1/')
    echo "__init__.py version: $INIT_PY_VERSION"
else
    echo -e "${RED}Error: __init__.py not found at $INIT_PY_PATH${NC}"
    exit 1
fi

# Check if versions match
if [ "$PYPROJECT_VERSION" != "$VERSION_PY_VERSION" ] || [ "$PYPROJECT_VERSION" != "$INIT_PY_VERSION" ]; then
    echo -e "${RED}Error: Version mismatch detected!${NC}"
    echo "pyproject.toml: $PYPROJECT_VERSION"
    echo "version.py: $VERSION_PY_VERSION"
    echo "__init__.py: $INIT_PY_VERSION"
    
    read -p "Do you want to update all versions to match pyproject.toml ($PYPROJECT_VERSION)? (y/N) " UPDATE_CONFIRM
    if [[ "$UPDATE_CONFIRM" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Updating versions to $PYPROJECT_VERSION...${NC}"
        
        # Update version.py
        sed -i.bak "s/__version__ = \"$VERSION_PY_VERSION\"/__version__ = \"$PYPROJECT_VERSION\"/" "$VERSION_PY_PATH"
        rm -f "$VERSION_PY_PATH.bak"
        
        # Update __init__.py
        sed -i.bak "s/__version__ = \"$INIT_PY_VERSION\"/__version__ = \"$PYPROJECT_VERSION\"/" "$INIT_PY_PATH"
        rm -f "$INIT_PY_PATH.bak"
        
        echo -e "${GREEN}Versions updated to $PYPROJECT_VERSION${NC}"
    else
        echo -e "${RED}Version mismatch not resolved. Exiting.${NC}"
        exit 1
    fi
fi

# Clean previous builds
echo -e "${GREEN}Cleaning previous builds...${NC}"
rm -rf "$PROJECT_ROOT/dist" "$PROJECT_ROOT/build" "$PROJECT_ROOT/*.egg-info"

# Build the package
echo -e "${GREEN}Building package...${NC}"
cd "$PROJECT_ROOT"
python3 -m build

# Check the built package
echo -e "${GREEN}Built package files:${NC}"
ls -la "$PROJECT_ROOT/dist"

# Confirm upload
echo -e "${YELLOW}Do you want to upload the package to PyPI? (y/N)${NC}"
read -r UPLOAD_CONFIRM

if [[ "$UPLOAD_CONFIRM" =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Uploading to PyPI...${NC}"
    python3 -m twine upload "$PROJECT_ROOT/dist/"*
    echo -e "${GREEN}Package version $PYPROJECT_VERSION uploaded to PyPI${NC}"
else
    echo -e "${YELLOW}Skipping upload. Package is built in the dist directory.${NC}"
    echo -e "${YELLOW}To upload manually, run: python -m twine upload dist/*${NC}"
fi

echo -e "${GREEN}=== Build process completed ===${NC}"