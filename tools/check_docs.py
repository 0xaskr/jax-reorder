"""检查仓库 Markdown 的本地链接；--changed 仅检查相对 HEAD 的变更与新文件。"""

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")


def git_paths(*args):
    return subprocess.check_output(["git", *args, "-z"], cwd=ROOT).decode().split("\0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changed", action="store_true")
    args = parser.parse_args()
    tracked = (
        git_paths("diff", "HEAD", "--name-only", "--diff-filter=ACMR")
        if args.changed
        else git_paths("ls-files")
    )
    names = sorted(set(tracked + git_paths("ls-files", "--others", "--exclude-standard")))
    files = links = 0
    errors = []
    for name in names:
        path = ROOT / name
        if path.suffix != ".md" or not path.is_file() or name.startswith((".agents/", ".codex/")):
            continue
        files += 1
        # 代码围栏中展示的文档模板不属于当前页面链接。
        source = re.sub(r"```.*?```", "", path.read_text(), flags=re.DOTALL)
        for match in LINK.finditer(source):
            target = match.group(1).strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            links += 1
            local = unquote(parsed.path)
            resolved = ROOT / local.lstrip("/") if local.startswith("/") else path.parent / local
            if not resolved.exists():
                errors.append(f"{name}: missing local target {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Markdown local links: {files} files, {links} targets checked")


if __name__ == "__main__":
    main()
