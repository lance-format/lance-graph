#!/usr/bin/env python3
"""
Script to calculate the next version based on release type
"""

import argparse
import os
import sys
from packaging import version


def calculate_next_version(current_version, release_type, channel):
    """Calculate the next version based on release type and channel"""

    # Parse current version
    v = version.parse(current_version)

    # Extract major, minor, patch
    if hasattr(v, "release"):
        major, minor, patch = (
            v.release[:3] if len(v.release) >= 3 else (*v.release, 0, 0)[:3]
        )
    else:
        # Fallback for simple versions
        parts = current_version.split(".")
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0

    # Calculate new version based on release type
    # Note: 'current' keeps the same version (used to finalize previews to stable)
    if release_type == "major":
        new_version = f"{major + 1}.0.0"
    elif release_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    elif release_type == "patch":
        new_version = f"{major}.{minor}.{patch + 1}"
    elif release_type == "current":
        # Keep current version - used for:
        # - Subsequent preview releases (v0.0.16-beta.2, beta.3, etc.)
        # - Finalizing preview to stable (v0.0.16-beta.X -> v0.0.16)
        new_version = current_version
    else:
        raise ValueError(f"Unknown release type: {release_type}")

    return new_version


def main():
    parser = argparse.ArgumentParser(description="Calculate next version")
    parser.add_argument("--current", required=True, help="Current version")
    parser.add_argument(
        "--type",
        required=True,
        choices=["major", "minor", "patch", "current"],
        help="Release type",
    )
    parser.add_argument(
        "--channel",
        required=True,
        choices=["stable", "preview"],
        help="Release channel",
    )

    args = parser.parse_args()

    try:
        new_version = calculate_next_version(args.current, args.type, args.channel)

        # Output for GitHub Actions
        print(f"version={new_version}")

        # Also write to GITHUB_OUTPUT if available
        github_output = os.environ.get("GITHUB_OUTPUT")
        if github_output:
            with open(github_output, "a") as f:
                f.write(f"version={new_version}\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
