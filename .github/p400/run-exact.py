import hashlib, json, pathlib, re, subprocess, sys
binary = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
requested = sys.argv[3:]
assert requested and len(requested) == len(set(requested))
listing = subprocess.run([str(binary), "--list"], check=True, capture_output=True, text=True, timeout=90).stdout
(out / "TESTS.txt").write_text(listing)
names = [line.removesuffix(": test") for line in listing.splitlines() if line.endswith(": test")]
for name in requested:
    assert names.count(name) == 1, (name, names.count(name))
results = []
for name in requested:
    command = [str(binary), name, "--exact", "--nocapture", "--test-threads=1", "--color", "never"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        code, output = result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired as error:
        code = 124
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")
        output = decode(error.stdout) + decode(error.stderr) + "\nTIMEOUT after 90 seconds\n"
    good = code == 0 and re.search(r"test result: ok\. 1 passed; 0 failed;", output) is not None
    (out / (name.rsplit("::", 1)[-1] + ".log")).write_text(output)
    results.append({"test": name, "command": command, "returncode": code, "one_test_passed": good})
receipt = {"harness_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(), "results": results}
(out / "RESULT.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
sys.exit(0 if all(item["one_test_passed"] for item in results) else 1)
