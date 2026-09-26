"""
generate_placeholders.py -- reads isro_kb.json and refreshes README.md's
image list (which entities have a file, which still need one). For an
entity whose "image" names a file that doesn't exist yet, it also writes a
plain grey placeholder SVG so the path resolves; entities with
"image": null are skipped -- the UI renders its own styled panel for those.
Safe to run repeatedly -- never overwrites a real image.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KB_PATH = HERE.parent.parent.parent / "isro_kb.json"
README_PATH = HERE / "README.md"

SVG_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400" viewBox="0 0 640 400">
  <rect width="640" height="400" fill="#1c2333"/>
  <rect x="1" y="1" width="638" height="398" fill="none" stroke="#3a4360" stroke-width="2"/>
  <text x="320" y="190" font-family="sans-serif" font-size="28" fill="#8fa3c7" text-anchor="middle">{name}</text>
  <text x="320" y="225" font-family="sans-serif" font-size="14" fill="#5c6a8c" text-anchor="middle">placeholder image</text>
</svg>
"""


def main() -> None:
    entities = json.loads(KB_PATH.read_text(encoding="utf-8"))

    created, skipped = [], []
    for ent in entities:
        filename = ent["image"]
        if not filename:
            continue  # no image yet -- the UI shows a styled name/status panel
        path = HERE / filename
        if path.exists():
            skipped.append(filename)
            continue
        path.write_text(SVG_TEMPLATE.format(name=ent["name"]), encoding="utf-8")
        created.append(filename)

    print(f"created {len(created)} placeholder(s), skipped {len(skipped)} existing file(s)")

    lines = [f"- `{ent['image']}` -- {ent['name']}" if ent["image"]
             else f"- **needs image** -- {ent['name']} (`{ent['id']}`)" for ent in entities]
    readme = README_PATH.read_text(encoding="utf-8")
    marker = "<!-- FILENAME_LIST -->"
    readme = readme.split(marker)[0] + marker + "\n" + "\n".join(lines) + "\n"
    README_PATH.write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
