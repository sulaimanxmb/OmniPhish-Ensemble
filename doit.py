from pathlib import Path
import shutil

# Source directory containing .py files
SOURCE_DIR = Path(".")

# Output directory for generated .md files
OUTPUT_DIR = Path("converted_md_files")
OUTPUT_DIR.mkdir(exist_ok=True)

for py_file in SOURCE_DIR.rglob("*.py"):
    # Skip files already inside output directory
    if OUTPUT_DIR in py_file.parents:
        continue

    # Preserve folder structure
    relative_path = py_file.relative_to(SOURCE_DIR)
    md_file = OUTPUT_DIR / relative_path.with_suffix(".md")

    md_file.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(py_file, md_file)

print("Conversion complete!")