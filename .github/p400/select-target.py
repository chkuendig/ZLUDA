import json,os,pathlib
assert os.environ.get("RUNNER_ENVIRONMENT")=="github-hosted","Cloud hosted runners only"
pins=json.loads(pathlib.Path(__file__).with_name("pins.json").read_text())
choice=os.environ["TARGET"]
choices={"all":["jc2","reference","trace"],"native-both":["jc2","reference"],**{k:[k] for k in pins}}
assert choice in choices,"Target is not in the public allowlist"
print("matrix="+json.dumps({"include":[dict(id=k,commit=pins[k]["commit"],llvm=pins[k]["llvm_gitlink"]) for k in choices[choice]]}))
