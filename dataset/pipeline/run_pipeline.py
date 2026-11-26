import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from engine import Episode, SUPPORTED_CONDITIONS, SUPPORTED_METHODS

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent

def plans(seed, count):
    import random
    import sys
    sys.path.insert(0,str(DATA/"pools"))
    from sample_catalog import choose_methods
    rng=random.Random(seed)
    catalog=yaml.safe_load((DATA/"pools/catalog.yaml").read_text())
    filtered={**catalog,"methods":{m:catalog["methods"][m] for m in SUPPORTED_METHODS}}
    card_path=DATA/"agent_tuning/tuned/answer.json"
    card=next(item for item in json.loads(card_path.read_text())["cases"] if item["id"]=="grasp")
    assert card["probe_negative_ends_task"] is False
    assert card["branches"]["negative"]["next_action"]=="regrasp_current_object"
    physics_keys={"C01":"friction_multiplier","C03":"mass_multiplier","C04":"grasp_offset_m"}
    result=[]
    for index in range(count):
        task=rng.choice([0,1,2])
        demo=rng.randrange(20)
        size=rng.choices([1,2,3],weights=[50,40,10],k=1)[0]
        conditions=["C01"]+rng.sample(["C03","C04"],size-1)
        # 同一源演示的首测固定，不根据隐藏物理参数选择。
        first=SUPPORTED_METHODS[(task*50+demo)%len(SUPPORTED_METHODS)]
        selected=choose_methods(tuple(conditions),"grasp_check",first,filtered,rng)
        physics={"friction_multiplier":1.0,"mass_multiplier":1.0,"grasp_offset_m":0.0}
        for cid in conditions:
            choices=catalog["conditions"][cid]["candidates"]
            choices=[x for x in choices if x != (0.0 if cid=="C04" else 1.0)]
            physics[physics_keys[cid]]=rng.choice(choices)
        result.append({
            "plan_id":f"selftest_{index:03d}","task_id":task,"demo_id":demo,"seed":seed+index,
            "physics":physics,"condition_ids":conditions,
            "first_method":first,"second_method":selected[1],
            "candidate_methods":selected,"feedback":True,
            "source_kind":"original_libero_expert","split_group":f"spatial_{task}_demo_{demo}",
            "planner_source":"reviewed_deepseek_branch_card_and_seeded_catalog_sampling",
            "task_card":str(card_path),"public_stage":"grasp_check",
        })
    return result

def run_batch(output, seed, count, resume=False):
    if resume:
        chosen=json.loads((output/"plans.json").read_text())
        assert len(chosen)==count,"续跑需要使用已保存计划的实际数量"
    else:
        output.mkdir(parents=True,exist_ok=False)
        chosen=plans(seed,count)
        (output/"plans.json").write_text(json.dumps(chosen,ensure_ascii=False,indent=2))
    rows = []
    for plan in chosen:
        destination=output/plan["plan_id"]
        metadata_path=destination/"metadata.json"
        if resume and metadata_path.exists():
            record=json.loads(metadata_path.read_text())
            assert record["plan"]==plan and (destination/"trajectory.hdf5").is_file()
        else:
            episode = Episode(plan)
            try:
                record = episode.run()
                episode.save(destination,record)
            finally:
                episode.close()
        record["directory"]=str(destination)
        rows.append(record)
        summary = {
            "attempted":len(rows),"successes":sum(row["success"] for row in rows),
            "negative_probe_episodes":sum(any(e["result"]=="negative" for e in row["events"]) for row in rows),
            "successful_negative_probe_episodes":sum(row["success"] and any(e["result"]=="negative" for e in row["events"]) for row in rows),
            "methods":dict(Counter(event["method"] for row in rows for event in row["events"] if event["method"].startswith("P"))),
            "records":rows,
        }
        (output/"manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        print("BATCH_PROGRESS",json.dumps({k:v for k,v in summary.items() if k!="records"},ensure_ascii=False),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--seed",type=int,default=20261010)
    parser.add_argument("--count",type=int,default=12)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--render",action="store_true")
    args=parser.parse_args()
    assert args.count>=1
    summary=run_batch(args.output,args.seed,args.count,args.resume)
    subprocess.run([sys.executable,str(ROOT/"audit_dataset.py"),"--batch",str(args.output)],check=True)
    if args.render:
        successes=[row for row in summary["records"] if row["success"]]
        recovered=[row for row in successes if any(e["result"]=="negative" for e in row["events"])]
        selected=successes[:1]+[row for row in recovered[:1] if row not in successes[:1]]
        for row in selected:
            episode=args.output/row["plan"]["plan_id"]
            gallery=args.output/("gallery_"+row["plan"]["plan_id"])
            subprocess.run([
                sys.executable,str(ROOT/"render_robot.py"),
                "--trajectory",str(episode/"trajectory.hdf5"),"--xml",str(episode/"model.xml"),
                "--output",str(gallery),"--orbit","--stride","2",
            ],check=True)
    print("PIPELINE_COMPLETE",json.dumps({k:v for k,v in summary.items() if k!="records"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
