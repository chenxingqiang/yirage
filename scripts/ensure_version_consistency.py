#!/usr/bin/env python3
"""
Ensure version consistency across the package before building
"""

import os
import re
import sys

def read_version_from_file(filepath):
    """Read version from a Python file"""
    with open(filepath, 'r') as f:
        content = f.read()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
    return None

def update_version_in_file(filepath, old_version, new_version):
    """Update version in a Python file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Replace version
    updated_content = re.sub(
        r'__version__\s*=\s*["\']' + re.escape(old_version) + r'["\']',
        f'__version__ = "{new_version}"',
        content
    )
    
    with open(filepath, 'w') as f:
        f.write(updated_content)

def main():
    """Main function"""
    # Get the project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Paths to version files
    version_py_path = os.path.join(project_root, 'python', 'yirage', 'version.py')
    init_py_path = os.path.join(project_root, 'python', 'yirage', '__init__.py')
    
    # Read versions
    version_py_version = read_version_from_file(version_py_path)
    init_py_version = read_version_from_file(init_py_path)
    
    if not version_py_version:
        print(f"Error: Could not find version in {version_py_path}")
        sys.exit(1)
    
    if not init_py_version:
        print(f"Error: Could not find version in {init_py_path}")
        sys.exit(1)
    
    # Check if versions match
    if version_py_version == init_py_version:
        print(f"✅ Versions match: {version_py_version}")
        return
    
    # Update version in __init__.py
    print(f"⚠️  Version mismatch: version.py={version_py_version}, __init__.py={init_py_version}")
    print(f"📝 Updating __init__.py to version {version_py_version}...")
    update_version_in_file(init_py_path, init_py_version, version_py_version)
    print(f"✅ Updated __init__.py to version {version_py_version}")

if __name__ == "__main__":
    main()
