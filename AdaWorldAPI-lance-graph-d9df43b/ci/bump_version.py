#!/usr/bin/env python3
"""
Version management script for lance-graph project.
Uses bump-my-version to handle version bumping across all project components.

Versioning scheme:
  - Stable releases: X.Y.Z (e.g., 0.1.0, 1.0.0)
  - Pre-releases: X.Y.Z-<label>.<n> (e.g., 0.1.0-beta.1, 0.2.0-rc.2)
  - Labels: alpha, beta, rc

Examples:
  # Bump to a specific version
  python ci/bump_version.py --version 0.2.0
  python ci/bump_version.py --version 0.2.0-beta.1

  # Bump a specific part
  python ci/bump_version.py --bump patch       # 0.1.0 -> 0.1.1
  python ci/bump_version.py --bump minor       # 0.1.0 -> 0.2.0
  python ci/bump_version.py --bump major       # 0.1.0 -> 1.0.0
  python ci/bump_version.py --bump pre_n       # 0.1.0-beta.1 -> 0.1.0-beta.2
  python ci/bump_version.py --bump pre_label   # 0.1.0-beta.1 -> 0.1.0-rc.1

  # Dry run to see changes without applying them
  python ci/bump_version.py --version 0.2.0 --dry-run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture_output, text=True)
    if result.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        if capture_output:
            print(f"stderr: {result.stderr}")
        sys.exit(result.returncode)
    return result


def get_current_version() -> str:
    """Get the current version from .bumpversion.toml."""
    config_path = Path(".bumpversion.toml")
    if not config_path.exists():
        raise FileNotFoundError(".bumpversion.toml not found in current directory")

    with open(config_path, "r") as f:
        for line in f:
            if line.strip().startswith('current_version = "'):
                return line.split('"')[1]
    raise ValueError("Could not find current_version in .bumpversion.toml")


def parse_version(version: str) -> dict:
    """Parse version string into components."""
    pattern = r"(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(-(?P<pre_label>alpha|beta|rc)\.(?P<pre_n>\d+))?"
    match = re.match(pattern, version)
    if not match:
        raise ValueError(f"Invalid version format: {version}")
    return match.groupdict()


def is_prerelease(version: str) -> bool:
    """Check if version is a pre-release version."""
    parts = parse_version(version)
    return parts.get("pre_label") is not None


def main():
    parser = argparse.ArgumentParser(
        description='Bump version using bump-my-version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--version', help='New version to set (e.g., 0.2.0 or 0.2.0-beta.1)')
    group.add_argument('--bump', choices=['major', 'minor', 'patch', 'pre_label', 'pre_n'],
                       help='Bump a specific version part')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be changed without making changes')

    args = parser.parse_args()

    # Get current version
    current_version = get_current_version()
    print(f"Current version: {current_version}")

    # Build base command
    base_cmd = ["bump-my-version", "bump"]

    if args.dry_run:
        print("\nDry run mode - no changes will be made")
        base_cmd.extend(["--dry-run", "--verbose"])
    else:
        base_cmd.extend(["--no-commit", "--no-tag"])

    base_cmd.append("--allow-dirty")

    if args.version:
        # Explicit version specified
        new_version = args.version
        print(f"Target version: {new_version}")

        # Validate version format
        try:
            parse_version(new_version)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

        cmd = base_cmd + ["--current-version", current_version, "--new-version", new_version]
    else:
        # Bump a specific part
        bump_part = args.bump

        # Validate that pre_n and pre_label bumps are valid for current version
        if bump_part in ["pre_n", "pre_label"] and not is_prerelease(current_version):
            print(f"Error: Cannot bump '{bump_part}' on a stable version ({current_version}).")
            print("Use --version to set a specific pre-release version first.")
            sys.exit(1)

        cmd = base_cmd + [bump_part]

    run_command(cmd, capture_output=False)

    if not args.dry_run:
        new_version = get_current_version()
        print(f"\nSuccessfully updated version from {current_version} to {new_version}")


if __name__ == '__main__':
    main()
