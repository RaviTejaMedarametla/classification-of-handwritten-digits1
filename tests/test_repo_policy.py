import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RepositoryPolicyTests(unittest.TestCase):
    def test_no_hyperskill_dependencies(self):
        banned = ("hs" + "-test-python", "hs" + "test", "Hyper" + "skill", "Stage" + "Test", "Check" + "Result")
        for path in ROOT.rglob("*"):
            if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in banned:
                self.assertNotIn(token, text, msg=f"Found {token} in {path}")

    def test_requirements_are_ci_focused(self):
        req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("numpy==", req)
        self.assertIn("pandas==", req)
        self.assertIn("matplotlib==", req)
        self.assertIn("scikit-learn==", req)
        self.assertIn("psutil==", req)

    def test_ci_uses_python_310_and_cache(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("python-version: '3.10'", ci)
        self.assertIn("cache: 'pip'", ci)


if __name__ == "__main__":
    unittest.main()
