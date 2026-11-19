import re
import subprocess
from pathlib import Path
from unittest import TestCase
from urllib.parse import unquote

ROOT = Path(__file__).parents[1]


def tracked(pattern: str) -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", pattern], cwd=ROOT, text=True)
    return [ROOT / name for name in output.splitlines()]


class DocumentationTests(TestCase):
    def test_skill_frontmatter_matches_directory(self) -> None:
        root = ROOT / "ai/skills"
        skills = sorted(path for path in root.iterdir() if path.is_dir())
        self.assertEqual(
            {path.name for path in skills},
            {"make_report", "make_monthly_report", "run_experiment"},
        )
        for directory in skills:
            text = (directory / "SKILL.md").read_text()
            self.assertTrue(text.startswith("---\n"), directory.name)
            _, header, _ = text.split("---", 2)
            fields = dict(line.split(":", 1) for line in header.strip().splitlines())
            self.assertEqual(fields["name"].strip(), directory.name)
            self.assertTrue(fields["description"].strip())

    def test_local_markdown_links_exist(self) -> None:
        missing = []
        for source in tracked("*.md"):
            text = source.read_text()
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                target = target.strip()
                if not target or target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                target = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if target and not (source.parent / target).resolve().exists():
                    missing.append(f"{source.relative_to(ROOT)} -> {target}")
        self.assertEqual(missing, [])

    def test_active_docs_do_not_use_removed_entry_points(self) -> None:
        stale = ("configs/flows/", "runtime/train.py", "runtime/inference.py")
        archive = ROOT / "docs/euclid_q1/README.md"
        violations = []
        for source in tracked("*.md"):
            if source == archive:
                continue
            text = source.read_text()
            for value in stale:
                if value in text:
                    violations.append(f"{source.relative_to(ROOT)}: {value}")
        self.assertEqual(violations, [])

    def test_agent_import_and_monthly_placeholders(self) -> None:
        self.assertEqual((ROOT / "CLAUDE.md").read_text().strip(), "@AGENTS.md")
        for name in ("template.tex", "plan_template.tex"):
            text = (ROOT / "ai/skills/make_monthly_report" / name).read_text()
            self.assertIn("<<НАУЧНЫЕ РУКОВОДИТЕЛИ>>", text)
            self.assertNotIn("<научные руководители>", text)

    def test_no_tracked_desktop_database_files(self) -> None:
        self.assertFalse(any(path.name == ".DS_Store" for path in tracked("ai/**")))
