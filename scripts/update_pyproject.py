from __future__ import annotations

import re
import sys
from pathlib import Path


def update_pyproject(path: Path) -> None:
    if not path.exists():
        path.write_text(
            """[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aereo-water-segmentation"
version = "3.0.0"
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
include = ["waterseg*", "aereo_water*"]
""",
            encoding="utf-8",
        )
        return

    text = path.read_text(encoding="utf-8")
    if "[tool.setuptools.packages.find]" in text:
        section_start = text.index("[tool.setuptools.packages.find]")
        next_section = text.find("\n[", section_start + 1)
        section_end = next_section if next_section != -1 else len(text)
        section = text[section_start:section_end]
        include_match = re.search(
            r"(?m)^include\s*=\s*\[(.*?)\]\s*$",
            section,
        )
        if include_match:
            values = re.findall(r'["\']([^"\']+)["\']', include_match.group(1))
            for value in ("waterseg*", "aereo_water*"):
                if value not in values:
                    values.append(value)
            replacement = "include = [" + ", ".join(
                f'"{value}"' for value in values
            ) + "]"
            section = (
                section[: include_match.start()]
                + replacement
                + section[include_match.end() :]
            )
            text = text[:section_start] + section + text[section_end:]
        elif "where" not in section:
            section += '\nwhere = ["src"]\ninclude = ["waterseg*", "aereo_water*"]\n'
            text = text[:section_start] + section + text[section_end:]
        path.write_text(text, encoding="utf-8")
        return

    if re.search(r"(?m)^packages\s*=", text):
        raise RuntimeError(
            "pyproject.toml uses explicit package declarations. Replace them "
            "with setuptools package discovery:\n\n"
            '[tool.setuptools.packages.find]\nwhere = ["src"]\n'
            'include = ["waterseg*", "aereo_water*"]\n'
        )

    text = text.rstrip() + (
        '\n\n[tool.setuptools.packages.find]\n'
        'where = ["src"]\n'
        'include = ["waterseg*", "aereo_water*"]\n'
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pyproject.toml")
    update_pyproject(target)
    print(f"Updated package discovery in {target}")
