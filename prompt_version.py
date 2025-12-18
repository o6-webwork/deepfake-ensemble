#!/usr/bin/env python3
"""
Prompt Version Management Tool

This script helps manage prompt versions in prompts.yaml, including:
- Viewing current version
- Viewing changelog
- Creating new versions
- Comparing versions (when multiple versions exist)

Usage:
    python prompt_version.py info           # Show current version info
    python prompt_version.py changelog      # Show full changelog
    python prompt_version.py bump [major|minor|patch]  # Bump version
"""

import yaml
import sys
from datetime import datetime
from pathlib import Path


class PromptVersionManager:
    """Manage prompt versions in prompts.yaml"""

    def __init__(self, prompts_path="prompts.yaml"):
        self.prompts_path = Path(prompts_path)
        if not self.prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

        with open(self.prompts_path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)

        self.metadata = self.data.get('metadata', {})

    def get_version(self):
        """Get current version string."""
        return self.metadata.get('version', 'unknown')

    def get_changelog(self):
        """Get changelog entries."""
        return self.metadata.get('changelog', [])

    def show_info(self):
        """Display current version information."""
        print("=" * 60)
        print("NexInspect Prompt Version Information")
        print("=" * 60)
        print(f"Version:      {self.metadata.get('version', 'unknown')}")
        print(f"Last Updated: {self.metadata.get('last_updated', 'unknown')}")
        print(f"Description:  {self.metadata.get('description', 'N/A')}")
        print(f"File:         {self.prompts_path}")
        print()

        # Show latest changelog entry
        changelog = self.get_changelog()
        if changelog:
            latest = changelog[0]
            print("Latest Changes:")
            print(f"  Version: {latest.get('version', 'unknown')}")
            print(f"  Date:    {latest.get('date', 'unknown')}")
            print(f"  Author:  {latest.get('author', 'unknown')}")
            print("  Changes:")
            for change in latest.get('changes', []):
                print(f"    - {change}")
        print("=" * 60)

    def show_changelog(self):
        """Display full changelog."""
        print("=" * 60)
        print("Prompt Version Changelog")
        print("=" * 60)

        changelog = self.get_changelog()
        if not changelog:
            print("No changelog entries found.")
            return

        for entry in changelog:
            print(f"\nVersion {entry.get('version', 'unknown')} - {entry.get('date', 'unknown')}")
            print(f"Author: {entry.get('author', 'unknown')}")
            print("Changes:")
            for change in entry.get('changes', []):
                print(f"  - {change}")
            print("-" * 60)

    def bump_version(self, bump_type='patch', changes=None, author='User'):
        """
        Bump the version number and add changelog entry.

        Args:
            bump_type: 'major', 'minor', or 'patch'
            changes: List of change descriptions
            author: Author name
        """
        current_version = self.get_version()

        if current_version == 'unknown':
            print("Error: No current version found in metadata")
            return

        # Parse version
        try:
            major, minor, patch = map(int, current_version.split('.'))
        except ValueError:
            print(f"Error: Invalid version format: {current_version}")
            return

        # Bump version
        if bump_type == 'major':
            major += 1
            minor = 0
            patch = 0
        elif bump_type == 'minor':
            minor += 1
            patch = 0
        elif bump_type == 'patch':
            patch += 1
        else:
            print(f"Error: Invalid bump type: {bump_type}")
            return

        new_version = f"{major}.{minor}.{patch}"
        today = datetime.now().strftime('%Y-%m-%d')

        # Create new changelog entry
        new_entry = {
            'version': new_version,
            'date': today,
            'changes': changes or ['Version bump'],
            'author': author
        }

        # Update metadata
        self.metadata['version'] = new_version
        self.metadata['last_updated'] = today

        # Prepend to changelog
        changelog = self.get_changelog()
        changelog.insert(0, new_entry)
        self.metadata['changelog'] = changelog

        # Update data
        self.data['metadata'] = self.metadata

        # Write back to file
        with open(self.prompts_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(self.data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"✓ Bumped version: {current_version} → {new_version}")
        print(f"  Date: {today}")
        print(f"  Author: {author}")
        print("  Changes:")
        for change in new_entry['changes']:
            print(f"    - {change}")


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    manager = PromptVersionManager()

    if command == 'info':
        manager.show_info()

    elif command == 'changelog':
        manager.show_changelog()

    elif command == 'bump':
        bump_type = sys.argv[2] if len(sys.argv) > 2 else 'patch'
        changes = sys.argv[3:] if len(sys.argv) > 3 else None
        manager.bump_version(bump_type, changes)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()
