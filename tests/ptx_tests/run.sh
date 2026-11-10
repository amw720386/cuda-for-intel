#!/bin/sh
# vosen/ptx_tests on the SPIR-V simulator: every test's PTX goes through our translator
# (src/ptx.rs) and runs on src/ptx/sim.rs through simcuda, a 12-call Driver API library.
#   run.sh [--bless] [run.py options, e.g. --elements 4096 --filter '^cvt']
# Fetches ptx_tests (MIT OR Apache-2.0) at a pinned commit into target/, applies
# ptx_tests.patch (sampling and per-test panic isolation) and builds it with a nightly
# toolchain, or with stable and RUSTC_BOOTSTRAP=1 when no nightly is installed.
set -e
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
rev=76d4dcef6a2e40fb3ab173c2ceda5f0098218f8b
up="${PTX_TESTS_DIR:-$root/target/ptx_tests-upstream}"
if [ ! -d "$up/.git" ]; then
  git clone --quiet https://github.com/vosen/ptx_tests "$up"
fi
patch="$here/ptx_tests.patch"
# Reuse the checkout (and its build) when it is the pinned commit with exactly the patch applied.
if [ "$(git -C "$up" rev-parse HEAD)" != "$rev" ] || ! git -C "$up" apply --reverse --check "$patch" 2>/dev/null \
   || [ "$(git -C "$up" diff | git hash-object --stdin)" != "$(git hash-object "$patch")" ]; then
  git -C "$up" fetch --quiet origin "$rev" 2>/dev/null || true
  git -C "$up" checkout --quiet --force "$rev"
  git -C "$up" apply "$patch"
fi
if rustup run nightly rustc --version >/dev/null 2>&1; then
  (cd "$up" && cargo +nightly build --release --quiet)
else
  (cd "$up" && RUSTUP_TOOLCHAIN=stable RUSTC_BOOTSTRAP=1 cargo build --release --quiet)
fi
cargo build --release --quiet --manifest-path "$here/simcuda/Cargo.toml"
lib="$here/simcuda/target/release/libsimcuda.dylib"
[ -f "$lib" ] || lib="$here/simcuda/target/release/libsimcuda.so"
bless=
if [ "$1" = "--bless" ]; then bless=--bless; shift; fi
exec python3 -I "$here/run.py" --ptx-tests "$up/target/release/ptx_tests" --cuda "$lib" \
  --out "$root/target/ptx_tests-sim.tsv" $bless "$@"
