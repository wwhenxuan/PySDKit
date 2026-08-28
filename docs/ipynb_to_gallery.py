"""Convert ``examples/**/*.ipynb`` to Sphinx-Gallery Python scripts.

Markdown cells become RST text blocks (module docstring + ``# %%`` comments).
Code cells are copied as-is after stripping the local ``sys.path`` boilerplate.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

SECTION_TITLES = {
    "acmd": "Adaptive Chirp Mode Decomposition",
    "deconvolution": "Blind deconvolution",
    "emd": "Empirical Mode Decomposition",
    "emd_variants": "EMD variants",
    "ewt": "Empirical Wavelet Transform",
    "faemd": "Fast and Adaptive EMD",
    "gdmd": "Generalized and variational nonlinear MD",
    "image": "Images and 2-D decompositions",
    "imd": "Impulsive and polymorphic MD",
    "jmd": "Jump mode decomposition",
    "lmd": "Local Mean Decomposition",
    "memd": "Multivariate EMD",
    "osd": "Optimization-based decomposition",
    "ssa": "Singular Spectrum Analysis",
    "temp_iter": "Time-domain iterative methods",
    "tfa": "Time-frequency analysis",
    "tsa": "Time series decomposition",
    "utils": "Utilities",
    "vmd": "Variational Mode Decomposition",
    "vncmd": "Variational nonlinear chirp MD",
}

SECTION_BLURBS = {
    "acmd": "Chirp-mode methods that track instantaneous frequency with adaptive kernels.",
    "deconvolution": "Blind deconvolution tools for rotating-machinery fault signatures.",
    "emd": "Classical empirical mode decomposition, Hilbert–Huang analysis, and related demos.",
    "emd_variants": "Ensemble, robust, online, and filter-based extensions of EMD.",
    "ewt": "Empirical wavelet and empirical Fourier decompositions in 1-D and 2-D.",
    "faemd": "Fast and adaptive EMD for 1-D signals, images, and volumes.",
    "gdmd": "Generalized dispersive and variational nonlinear mode decompositions.",
    "image": "Bidimensional and multivariate decompositions for images.",
    "imd": "Impulsive and polymorphic mode decompositions.",
    "jmd": "Jump plus AM-FM mode decompositions.",
    "lmd": "Local mean decomposition and its robust variant.",
    "memd": "Multivariate EMD, adaptive projections, and MEMD filter banks.",
    "osd": "Optimization-based signal decomposition with proximal operators.",
    "ssa": "Singular spectrum analysis for trend / oscillatory extraction.",
    "temp_iter": "Iterative time-domain methods: ALIF, FMD, HVD, and ITD.",
    "tfa": "Synchrosqueezing, synchroextracting, and related time-frequency tools.",
    "tsa": "Seasonal-trend procedures (STL, MSTL, moving-average decomposition).",
    "utils": "Kurtogram, blind source separation, and synthetic-data helpers.",
    "vmd": "Variational mode decomposition and its successive / multivariate variants.",
    "vncmd": "Variational nonlinear chirp mode decomposition and adaptive relatives.",
}

HEADING_UNDERLINES = {1: "=", 2: "-", 3: "~", 4: "^", 5: '"'}

BOILERPLATE_RE = re.compile(
    r"^import sys\s*\n"
    r"from pathlib import Path\s*\n\s*"
    r"ROOT = Path\.cwd\(\)\.resolve\(\)\s*\n"
    r"if not \(ROOT / [\"']pysdkit[\"']\)\.is_dir\(\):\s*\n"
    r"    for parent in ROOT\.parents:\s*\n"
    r"        if \(parent / [\"']pysdkit[\"']\)\.is_dir\(\):\s*\n"
    r"            ROOT = parent\s*\n"
    r"            break\s*\n"
    r"if str\(ROOT\) not in sys\.path and \(ROOT / [\"']pysdkit[\"']\)\.is_dir\(\):\s*\n"
    r"    sys\.path\.insert\(0, str\(ROOT\)\)\s*\n?",
    re.MULTILINE,
)

PYSDKIT_PATH_PRINT_RE = re.compile(
    r"^print\(\s*[\"']pysdkit from:[\"']\s*,\s*Path\(__import__\([\"']pysdkit[\"']\)\.__file__\)"
    r"\.resolve\(\)\s*\)\s*\n?",
    re.MULTILINE,
)

DISPLAY_MATH_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)
INLINE_MATH_RE = re.compile(r"(?<!\$)\$(?!\$)(.+?)\$")
LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)]+)\)")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
FENCE_RE = re.compile(r"^```(\w*)\s*$")
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    return src.replace("\r\n", "\n").replace("\r", "\n")


def _underline(title: str, level: int) -> str:
    ch = HEADING_UNDERLINES.get(level, "^")
    return f"{title}\n{ch * max(len(title), 3)}"


def _rst_display_math(body: str) -> str:
    lines = ["", ".. math::", ""]
    lines.extend(f"   {line}" if line.strip() else "" for line in body.splitlines())
    lines.append("")
    return "\n".join(lines)


def _convert_inline(text: str) -> str:
    """Convert inline markdown (math, links, code, emphasis) to RST."""
    placeholders: list[str] = []

    def stash(value: str) -> str:
        placeholders.append(value)
        return f"\x00PH{len(placeholders) - 1}\x00"

    def inline_math(match: re.Match[str]) -> str:
        return stash(f":math:`{match.group(1).strip()}`")

    def image(match: re.Match[str]) -> str:
        alt, url = match.group(1), match.group(2)
        block = f"\n\n.. image:: {url}\n"
        if alt:
            block += f"   :alt: {alt}\n"
        return stash(block + "\n")

    def link(match: re.Match[str]) -> str:
        label, url = match.group(1), match.group(2)
        return stash(f"`{label} <{url}>`_")

    def inline_code(match: re.Match[str]) -> str:
        return stash(f"``{match.group(1)}``")

    text = IMAGE_RE.sub(image, text)
    text = LINK_RE.sub(link, text)
    text = INLINE_MATH_RE.sub(inline_math, text)
    text = INLINE_CODE_RE.sub(inline_code, text)
    text = BOLD_RE.sub(r"**\1**", text)
    text = ITALIC_RE.sub(r"*\1*", text)

    for i, value in enumerate(placeholders):
        text = text.replace(f"\x00PH{i}\x00", value)
    return text


def _split_table_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    cells: list[str] = []
    buf: list[str] = []
    in_math = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "$":
            in_math = not in_math
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < len(line) and line[i + 1] == "|" and in_math:
            buf.append("\\|")
            i += 2
            continue
        if ch == "|" and not in_math:
            cells.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    cells.append("".join(buf).strip())
    return cells


def _convert_table(lines: list[str]) -> str:
    rows = [_split_table_row(line) for line in lines if not TABLE_SEP_RE.match(line)]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    out = ["", ".. list-table::", "   :header-rows: 1", ""]
    for row in rows:
        cells = [_convert_inline(cell).replace("\n", " ") for cell in row]
        out.append(f"   * - {cells[0] if cells[0] else ' '}")
        for cell in cells[1:]:
            out.append(f"     - {cell}")
    out.append("")
    return "\n".join(out)


def _convert_blockquote(lines: list[str]) -> str:
    body = [re.sub(r"^\s*>\s?", "", line) for line in lines]
    converted = _convert_inline("\n".join(body)).strip()
    indented = ["    " + line if line.strip() else "" for line in converted.splitlines()]
    return "\n\n.. epigraph::\n\n" + "\n".join(indented) + "\n"


def _is_quote_continuation(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(("#", "|", ">", "-", "*", "`")):
        return False
    if FENCE_RE.match(stripped):
        return False
    if re.match(r"^\d+\.\s+", stripped):
        return False
    return True


def md_to_rst(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not md:
        return ""

    display_blocks: list[str] = []

    def _stash_display(match: re.Match[str]) -> str:
        display_blocks.append(match.group(1).strip())
        return f"\n\n@@DISPLAYMATH{len(display_blocks) - 1}@@\n\n"

    md = DISPLAY_MATH_RE.sub(_stash_display, md)
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        fence = FENCE_RE.match(stripped)
        if fence:
            lang = fence.group(1) or "text"
            i += 1
            code_lines: list[str] = []
            while i < n and not FENCE_RE.match(lines[i].strip()):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1
            out.append("")
            out.append(f".. code-block:: {lang}")
            out.append("")
            for code_line in code_lines:
                out.append("   " + code_line if code_line.strip() else "")
            out.append("")
            continue

        if stripped.startswith("|") and i + 1 < n and (
            TABLE_SEP_RE.match(lines[i + 1]) or lines[i + 1].strip().startswith("|")
        ):
            table_lines = [line]
            i += 1
            while i < n and (lines[i].strip().startswith("|") or TABLE_SEP_RE.match(lines[i])):
                table_lines.append(lines[i])
                i += 1
            out.append(_convert_table(table_lines))
            continue

        if stripped.startswith(">"):
            quote: list[str] = []
            while i < n:
                current = lines[i]
                if current.strip().startswith(">"):
                    quote.append(current)
                    i += 1
                    continue
                if quote and _is_quote_continuation(current):
                    quote.append("> " + current.lstrip())
                    i += 1
                    continue
                break
            out.append(_convert_blockquote(quote))
            continue

        heading = re.match(r"^(#{1,5})\s+(.*)$", stripped)
        if heading:
            level = len(heading.group(1))
            title = _convert_inline(heading.group(2).strip())
            out.append("")
            out.append(_underline(title, level))
            out.append("")
            i += 1
            continue

        if stripped in {"---", "***", "___"}:
            out.append("")
            i += 1
            continue

        bullet = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
        if bullet:
            indent, marker, rest = bullet.groups()
            rst_marker = "#. " if marker.endswith(".") else "* "
            extra_indent = " " * (len(indent) // 2 * 2)
            out.append(f"{extra_indent}{rst_marker}{_convert_inline(rest)}")
            i += 1
            continue

        out.append(_convert_inline(line))
        i += 1

    text = "\n".join(out)
    for idx, body in enumerate(display_blocks):
        text = text.replace(f"@@DISPLAYMATH{idx}@@", _rst_display_math(body).strip("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def rst_to_docstring(rst: str) -> str:
    body = rst.strip("\n")
    if '"""' in body:
        body = body.replace('"""', "'''")
    return f'r"""\n{body}\n"""\n'


def rst_to_comments(rst: str) -> str:
    lines = rst.strip("\n").splitlines()
    out = ["# %%"]
    for line in lines:
        out.append("#" if not line.strip() else f"# {line}")
    return "\n".join(out) + "\n"


def strip_boilerplate(code: str) -> str:
    code = BOILERPLATE_RE.sub("", code)
    code = PYSDKIT_PATH_PRINT_RE.sub("", code)
    leftover = code
    if "sys." not in leftover and not re.search(r"\bsys\b", leftover.replace("import sys", "")):
        leftover = re.sub(r"^import sys\s*\n", "", leftover, count=1, flags=re.MULTILINE)
    uses_path = bool(re.search(r"\bPath\b", leftover.replace("from pathlib import Path", "")))
    if not uses_path:
        leftover = re.sub(r"^from pathlib import Path\s*\n", "", leftover, count=1, flags=re.MULTILINE)
    leftover = re.sub(r"\n{3,}", "\n\n", leftover)
    return leftover.strip() + ("\n" if leftover.strip() else "")


def title_from_rst(rst: str, fallback: str) -> tuple[str, str]:
    """Ensure the RST block starts with a level-1 title. Return (title, rst)."""
    lines = rst.strip("\n").splitlines()
    if len(lines) >= 2 and set(lines[1]) == {"="} and len(lines[1]) >= 3:
        return lines[0].strip(), rst
    title = fallback.replace("_", " ").strip() or "Example"
    titled = _underline(title, 1) + "\n\n" + rst.lstrip()
    return title, titled


def convert_notebook(ipynb_path: Path) -> str:
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])
    fallback = ipynb_path.stem
    parts: list[str] = []
    wrote_docstring = False

    for cell in cells:
        kind = cell.get("cell_type")
        src = cell_source(cell).strip()
        if not src:
            continue

        if kind == "markdown":
            rst = md_to_rst(src)
            if not rst.strip():
                continue
            if not wrote_docstring:
                _, rst = title_from_rst(rst, fallback)
                parts.append(rst_to_docstring(rst))
                wrote_docstring = True
            else:
                parts.append(rst_to_comments(rst))
            continue

        if kind == "code":
            code = strip_boilerplate(src)
            if not code.strip():
                continue
            if not wrote_docstring:
                _, rst = title_from_rst("", fallback)
                parts.append(rst_to_docstring(rst))
                wrote_docstring = True
            if parts and not parts[-1].endswith("\n\n"):
                parts[-1] = parts[-1].rstrip() + "\n\n"
            parts.append(code if code.endswith("\n") else code + "\n")

    if not wrote_docstring:
        _, rst = title_from_rst("", fallback)
        parts.insert(0, rst_to_docstring(rst))

    text = "\n".join(parts).strip() + "\n"
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def write_gallery_header(path: Path, title: str, body: str) -> None:
    underline = "=" * len(title)
    path.write_text(f"{title}\n{underline}\n\n{body.rstrip()}\n", encoding="utf-8", newline="\n")


def convert_all(*, delete_ipynb: bool = True) -> list[Path]:
    written: list[Path] = []
    notebooks = sorted(EXAMPLES.rglob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No notebooks under {EXAMPLES}")

    write_gallery_header(
        EXAMPLES / "GALLERY_HEADER.rst",
        "Examples",
        "Each card is one algorithm demo executed by Sphinx-Gallery. "
        "Open a card for the theory, figures, and downloads of the Python "
        "source and a generated Jupyter notebook.",
    )

    for folder, title in SECTION_TITLES.items():
        section_dir = EXAMPLES / folder
        if not section_dir.is_dir():
            continue
        blurb = SECTION_BLURBS.get(folder, "")
        write_gallery_header(section_dir / "GALLERY_HEADER.rst", title, blurb)

    for ipynb in notebooks:
        if ".ipynb_checkpoints" in ipynb.parts:
            continue
        dest = ipynb.with_suffix(".py")
        dest.write_text(convert_notebook(ipynb), encoding="utf-8", newline="\n")
        written.append(dest)
        print(f"wrote {dest.relative_to(ROOT)}")
        if delete_ipynb:
            ipynb.unlink()

    return written


def main() -> None:
    keep = "--keep-ipynb" in sys.argv
    convert_all(delete_ipynb=not keep)


if __name__ == "__main__":
    main()
