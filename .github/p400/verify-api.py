import pathlib,re,sys
from capsule import PINS,load,write
root=pathlib.Path(sys.argv[1]);target=sys.argv[2];workflow=sys.argv[3]
assert re.fullmatch(r"[a-f0-9]{40}",workflow)
run=load(root/"RUN.json");artifact=load(root/"ARTIFACT.json");jobs=load(root/"JOBS.json")
assert run["repository"]["full_name"]=="chkuendig/ZLUDA"
assert run["head_repository"]["full_name"]=="chkuendig/ZLUDA"
assert run["head_branch"]=="ci/p400-capsules-20261107" and run["head_sha"]==workflow
assert run["path"]==".github/workflows/p400-capsules.yml"
assert run["event"] in ["push","workflow_dispatch"]
assert run["status"]=="completed" and run["conclusion"]=="success"
assert run["run_attempt"]==1,"Rerun requires new review"
assert artifact["workflow_run"]["id"]==run["id"] and artifact["workflow_run"]["head_sha"]==workflow
assert artifact["workflow_run"]["repository_id"]==run["repository"]["id"]
assert not artifact["expired"] and re.fullmatch(r"sha256:[a-f0-9]{64}",artifact.get("digest") or "")
expected=f"p400-{target}-{PINS[target]['commit']}-{run['id']}-{run['run_attempt']}"
assert artifact["name"]==expected and 0<artifact["size_in_bytes"]<=2*2**30
matching=[j for j in jobs["jobs"] if j["run_id"]==run["id"] and j["name"].startswith("build (") and target in j["name"].split(",")[0]]
assert len(matching)==1 and matching[0]["status"]=="completed" and matching[0]["conclusion"]=="success"
assert matching[0]["labels"]==["ubuntu-22.04"],matching[0]["labels"]
write(root/"API-ADMISSION.json",dict(repository="chkuendig/ZLUDA",target=target,run=run["id"],job=matching[0]["id"],workflow_commit=workflow,artifact=artifact["id"],digest=artifact["digest"]))
