"""检查仓库 Markdown 的本地链接；--changed 仅检查相对 HEAD 的变更与新文件。

运行：
  uv run python tools/check_docs.py            # 检查全部已跟踪 Markdown
  uv run python tools/check_docs.py --changed  # 仅检查变更/新增文件

只解析本地相对链接，跳过外部 URL、纯锚点与代码围栏中的示例；缺失目标时
抛出 SystemExit 并列出全部错误（退出码非零），否则打印统计并返回 0。
本检查不验证锚点存在、远程可达性或文档内容，也不改动任何文件。
"""

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]\n]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)")


def git_paths(*args):
    """调用 git 读取路径列表；-z 避免文件名空白被拆分，返回字符串列表。"""
    return subprocess.check_output(["git", *args, "-z"], cwd=ROOT).decode().split("\0")


def main():
    """解析参数，收集待检查文件并核对每个本地链接目标是否存在。

    无参数时检查全部已跟踪 Markdown；--changed 时只检查相对 HEAD 的变更
    与未跟踪新文件。副作用是打印统计或完整错误列表；有缺失链接时以非零
    退出码结束，全部存在时返回 0。
    """
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
        # 只检查仓库内 Markdown；忽略包含无关自动化元数据的隐藏目录。
        if path.suffix != ".md" or not path.is_file() or name.startswith((".agents/", ".codex/")):
            continue
        files += 1
        # 代码围栏中展示的文档模板不属于当前页面链接。
        source = re.sub(r"```.*?```", "", path.read_text(), flags=re.DOTALL)
        for match in LINK.finditer(source):
            target = match.group(1).strip("<>")
            parsed = urlsplit(target)
            # 外部 URL、mailto 与纯锚点不解析为本地文件。
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            links += 1
            local = unquote(parsed.path)
            # 以 / 开头的链接相对仓库根，其余相对当前 Markdown 所在目录。
            resolved = ROOT / local.lstrip("/") if local.startswith("/") else path.parent / local
            if not resolved.exists():
                errors.append(f"{name}: missing local target {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Markdown local links: {files} files, {links} targets checked")


if __name__ == "__main__":
    main()
