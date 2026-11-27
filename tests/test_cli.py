"""CLI remains an auxiliary, scriptable entry point for the diagnostics."""
import json
import os
import tempfile
import unittest

from tests._setup import FAST_SIZE
from material_matcher.cli import build_parser, main


class TestCliCoupling(unittest.TestCase):
    def test_coupling_subcommand_writes_report(self):
        outdir = tempfile.mkdtemp(prefix="mm_cli_")
        argv = ["coupling", "--size", str(FAST_SIZE), "--seed", "3",
                "--outdir", outdir]
        main(argv)
        path = os.path.join(outdir, "coupling.json")
        self.assertTrue(os.path.exists(path))
        with open(path) as fh:
            payload = json.load(fh)
        self.assertIn("equivalent_pair", payload)
        self.assertIn("stagnation", payload)
        # The degenerate family must never authorise a preset write.
        self.assertEqual(
            payload["equivalent_pair"]["preset_eligible"], "none")
        self.assertTrue(payload["surface_verdicts"])

    def test_parser_registers_all_commands(self):
        parser = build_parser()
        for cmd in ("demo", "fit", "render", "coupling"):
            actions = [a for a in parser._subparsers._group_actions
                       if hasattr(a, "choices")]
            names = set()
            for a in actions:
                names.update(a.choices.keys())
            self.assertIn(cmd, names)


if __name__ == "__main__":
    unittest.main()
