#!/usr/bin/env python3
"""Mac/Linux-runnable checks on the export tables: python3 -m unittest scripts/tests/test_api_audit.py"""
import collections, pathlib, re, subprocess, sys, unittest

root = pathlib.Path(__file__).resolve().parents[2]
FN = re.compile(r'#\[(?:unsafe\()?no_mangle\)?\]\s*(?:#\[[^\]]*\]\s*)*pub\s+(?:unsafe\s+)?extern\s+"[a-zA-Z-]+"\s+fn\s+(\w+)', re.S)
CRATES = ["src", "crates/cudart_shim/src", "crates/cublas_shim/src", "crates/nvml_shim/src",
          "crates/cufft_shim/src", "crates/cusparse_shim/src", "crates/cusolver_shim/src", "crates/nvrtc_shim/src"]


class ExportTables(unittest.TestCase):
    def test_no_symbol_is_exported_twice(self):
        for crate in CRATES:
            counts = collections.Counter()
            for path in (root / crate).rglob("*.rs"):
                counts.update(FN.findall(path.read_text()))
            self.assertEqual({k: v for k, v in counts.items() if v > 1}, {}, crate)

    def test_coverage_doc_is_current(self):
        result = subprocess.run([sys.executable, str(root / "scripts" / "api_audit.py"), "--check"])
        self.assertEqual(result.returncode, 0, "run: python3 scripts/api_audit.py")

    def test_stub_files_are_current(self):
        for kind, target in [("driver", "src/cuda/stubs.rs"), ("cudart", "crates/cudart_shim/src/stubs.rs"),
                             ("nvml", "crates/nvml_shim/src/stubs.rs"), ("cufft", "crates/cufft_shim/src/stubs.rs"),
                             ("cusparse", "crates/cusparse_shim/src/stubs.rs"), ("cusolver", "crates/cusolver_shim/src/stubs.rs")]:
            path = root / target
            before = path.read_text()
            subprocess.run([sys.executable, str(root / "scripts" / "gen_api_stubs.py"), kind], check=True, capture_output=True)
            after = path.read_text()
            self.assertEqual(before, after, f"{target} is stale: run scripts/gen_api_stubs.py {kind}")


if __name__ == "__main__":
    unittest.main()
