"""Build the side-by-side comparison viewer from compare_template.html and compare_data.json (../compare_export.py).

  python3 build_compare.py compare.html
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent

if __name__ == "__main__":
    page = (HERE / "compare_template.html").read_text().replace("__DATA__", (HERE / "compare_data.json").read_text())
    Path(sys.argv[1]).write_text(page)
    print(sys.argv[1], f"{len(page) / 1e6:.1f} MB")
