#!/usr/bin/env python3
"""Torch-free tests of the progressive harness logic (python3 scripts/torch/test_progressive.py)."""
import json
import os
import subprocess
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import progressive as P  # noqa: E402


class Done:
    def __init__(self, stdout="", stderr="", rc=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, rc


def result_line(**kw):
    return P.RESULT_TAG + json.dumps(kw)


class Harness(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(P.classify_error("CUDA error: no kernel image is available for execution"), "209")
        self.assertEqual(P.classify_error("ZE_RESULT_ERROR_DEVICE_LOST"), "device-lost")
        self.assertEqual(P.classify_error("boom"), "error")

    def test_first_divergence(self):
        self.assertIsNone(P.first_divergence([1, 2, 3], [1, 2, 3]))
        self.assertEqual(P.first_divergence([1, 2, 9], [1, 2, 3]), 2)
        self.assertEqual(P.first_divergence([1, 2], [1, 2, 3]), 2)

    def test_status(self):
        self.assertEqual(P.level_status([]), "fail")
        self.assertEqual(P.level_status([{"ok": True}, {"ok": False}]), "fail")
        self.assertEqual(P.level_status([{"ok": True}]), "pass")

    def test_parse_child_output(self):
        out = "noise\n" + result_line(status="pass") + "\n"
        self.assertEqual(P.parse_child_output(out), {"status": "pass"})
        self.assertIsNone(P.parse_child_output("nothing"))
        self.assertIsNone(P.parse_child_output(P.RESULT_TAG + "{bad"))

    def test_run_child_timeout_and_crash(self):
        def timeout(*a, **k):
            raise subprocess.TimeoutExpired(a[0], 1, stderr="tail")
        r = P.run_child("L2", 1, runner=timeout)
        self.assertEqual((r["status"], r["error_class"]), ("timeout", "timeout"))
        r = P.run_child("L2", 1, runner=lambda *a, **k: Done("", "device lost", 3))
        self.assertEqual((r["status"], r["error_class"], r["rc"]), ("error", "device-lost", 3))

    def test_l7_variants_and_matrix(self):
        calls = []

        def runner(cmd, **kw):
            v = kw["env"]["CFI_L7_VARIANT"]
            calls.append((v, kw["env"].get("CFI_REPORT_CC")))
            if v == "default":
                return Done(result_line(level="L7", status="error", error_class="209", error="no kernel image", checks=[], seconds=1))
            return Done(result_line(level="L7", status="pass", checks=[{"name": "x", "ok": True}], seconds=1))
        r = P.run_level("L7", 5, runner=runner)
        self.assertEqual(calls, [("default", None), ("cc37", "3.7"), ("math", None)])
        self.assertEqual(r["status"], "partial")
        md = P.matrix_markdown({"L7": r, "L2": {"status": "fail", "seconds": 2.0,
                                                  "checks": [{"name": "f16", "ok": False, "note": "max_abs=1"}]}}, {"torch": "2.2.2"})
        self.assertIn("default: error (209); cc37: pass; math: pass", md)
        self.assertIn("f16: max_abs=1", md)
        self.assertTrue(md.startswith("Host: torch=2.2.2"))

    def test_close_with_mock_torch(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        import torch
        a = torch.tensor([1.0, 2.0])
        self.assertTrue(P.close(a, a + 1e-7, 1e-5, 1e-6)[0])
        self.assertFalse(P.close(a, a + 1, 1e-5, 1e-6)[0])
        self.assertFalse(P.close(a, torch.tensor([1.0]), 1, 1)[0])


if __name__ == "__main__":
    unittest.main()
