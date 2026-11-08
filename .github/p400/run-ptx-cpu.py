import hashlib, json, os, pathlib, re, subprocess, sys

binary, out = map(pathlib.Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
skip = ["--skip", "_amdgpu", "--skip", "_cuda"]
listing = subprocess.run([str(binary), "--list", *skip], check=True,
                         capture_output=True, text=True, timeout=90).stdout
(out / "TESTS.txt").write_text(listing)
names = [line.removesuffix(": test") for line in listing.splitlines() if line.endswith(": test")]
assert names and len(names) == len(set(names))
assert not any("_amdgpu" in name or "_cuda" in name for name in names)
assert "test::spirv_run::legacy_slct_scalar_types" in names
command = [str(binary), *skip, "--nocapture", "--test-threads=1", "--color", "never"]
env = dict(os.environ, TEST_PTX_LLVM_FAIL_DIR=str(out / "llvm-failures"))
try:
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, env=env)
    code, output = result.returncode, result.stdout + result.stderr
except subprocess.TimeoutExpired as error:
    def decode(value):
        return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
    code = 124
    output = decode(error.stdout) + decode(error.stderr) + "\nTIMEOUT after 600 seconds\n"
(out / "suite.log").write_text(output)
match = re.search(r"test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;", output)
counts = dict(zip(["passed", "failed", "ignored"], map(int, match.groups()))) if match else None
good = (code == 0 and counts is not None and counts["passed"] > 0 and counts["failed"] == 0
        and counts["passed"] + counts["ignored"] == len(names))
receipt = dict(harness_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), command=command,
               selected_tests=names, skip_filters=["_amdgpu", "_cuda"], counts=counts,
               returncode=code, passed=good, timeout_seconds=600,
               llvm_fail_dir=env["TEST_PTX_LLVM_FAIL_DIR"])
(out / "RESULT.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
sys.exit(0 if good else 1)
