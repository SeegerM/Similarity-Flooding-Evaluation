"""Zero-dependency test runner, matching `harness/tests/run_tests.py`.

pytest is not installed in this checkout's environments; the tests themselves
are written to run unchanged under it once it is.

    harness/.venv-embed/Scripts/python.exe code/semstruct/tests/run_tests.py
"""

from __future__ import annotations

import inspect
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_semstruct  # noqa: E402


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(vars(test_semstruct).items())
             if name.startswith("test_") and callable(fn)]
    passed, failed = 0, []
    for name, fn in tests:
        tmp = None
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                tmp = Path(tempfile.mkdtemp(prefix="semstruct-test-"))
                fn(tmp_path=tmp)
            else:
                fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception:
            failed.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc()
        finally:
            if tmp is not None:
                shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{passed} passed, {len(failed)} failed out of {len(tests)}")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
