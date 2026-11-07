import json,os,pathlib,shutil,subprocess,sys
assert os.environ.get("RUNNER_ENVIRONMENT")=="github-hosted","Cloud hosted runners only"
free=shutil.disk_usage(".").free
receipt={"free_bytes":free,"required_bytes":50*2**30,"df":subprocess.check_output(["df","-B1","/"],text=True)}
pathlib.Path("cloud-disk.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2))
assert free>=50*2**30,"Cloud capacity insufficient; report before selecting a larger cloud runner"
