#!/usr/bin/env python3
"""Approved synthetic-only scope availability audit; no policy or new scores."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from tricompose_v12.report_scope_table import build_scope_table, SCHEMA, FINDINGS, EDGES
from prepare_blinded_report_review import checked_artifact, freeze_method, SCHEMA as PACKET_SCHEMA
from report_review_sheets import load_bundle
from repair_cached_report_evidence import scope_check
from contracts import (PROTECTED_ROOT,require_inside,read_json,sha256_file,new_atomic_run,
    commit_atomic_run,discard_atomic_run,write_private_json,write_private_text)


def load(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("approved Slurm required before synthetic report scope audit")
    frozen,methods=freeze_method()
    packet=require_inside(args.packet_run,PROTECTED_ROOT,must_exist=True)
    pm=read_json(packet/"manifest.json")
    if pm.get("schema_version")!=PACKET_SCHEMA:
        raise ValueError("approved report review packet required")
    resolver_path,_=checked_artifact(packet,"investigator/resolver.json")
    freeze_path,_=checked_artifact(packet,"investigator/method_freeze.json")
    if read_json(freeze_path)!=frozen:
        raise ValueError("source method freeze differs")
    resolver=read_json(resolver_path)["records"]
    by_id={row["candidate_id"]:row for row in resolver}
    if len(by_id)!=48 or len(resolver)!=48:
        raise ValueError("complete 48-candidate source inventory required")
    root=require_inside(args.finding_run,PROTECTED_ROOT,must_exist=True)
    fm=read_json(root/"manifest.json")
    if fm.get("schema_version")!="tricompose-cached-finding-review-v1" or fm.get("metadata_only") is not True:
        raise ValueError("immutable cached finding review required")
    path,_=checked_artifact(root,"fact_evidence.jsonl")
    if path.stat().st_size>4*1024*1024:
        raise ValueError("bounded cached finding table required")
    facts=[json.loads(line) for line in path.read_text().splitlines() if line]
    if len(facts)!=48*len(FINDINGS) or {row["triple_candidate_id"] for row in facts}!=set(by_id):
        raise ValueError("all eight findings for 48 candidates must remain")
    sources={"packet_manifest":packet/"manifest.json","resolver":resolver_path,"method_freeze":freeze_path,
        "finding_manifest":root/"manifest.json","fact_evidence":path,**methods}
    upstream={}
    for name in ("lineages","crossmodal_details"):
        source=require_inside(fm["source_paths"][name],PROTECTED_ROOT,must_exist=True)
        h=sha256_file(source)
        packet_name="selection_table" if name=="lineages" else name
        if h!=fm["source_sha256"][name] or h!=pm["source_sha256"][packet_name]:
            raise ValueError("finding and report packet source bindings differ")
        sources[name]=source;upstream[name]=source
    details=read_json(upstream["crossmodal_details"])
    if details.get("evaluation_scope")!={"cohort":"fully_synthetic","raw_source_target_supplied":False,
            "real_reference_report_supplied":False,"unknown_is_negative":False}:
        raise ValueError("fully synthetic source only")
    refs={row["report_candidate_id"]:row for row in details["records"]}
    lines=[json.loads(line) for line in upstream["lineages"].read_text().splitlines() if line]
    lineages={row["triple_candidate_id"]:row["lineage"] for row in lines}
    if set(refs)!=set(by_id) or set(lineages)!=set(by_id) or len(lines)!=48 or len(details["records"])!=48:
        raise ValueError("source lineage inventory differs")
    enriched=[]
    for row in facts:
        cid=row["triple_candidate_id"];original=by_id[cid];ref=refs[cid];lineage=lineages[cid]
        hashes=row["artifact_hashes"]
        if (row["report_candidate_id"]!=cid or row["case_id"]!=original["case_id"] or
                any(hashes[key]!=lineage[key] for key in hashes) or
                row["cxr_candidate_id"]!=lineage["cxr_candidate_id"] or
                hashes["report_sha256"]!=original["report_sha256"] or
                hashes["cxr_sha256"]!=original["cxr_sha256"] or
                hashes["ehr_sha256"]!=original["ehr_sha256"]):
            raise ValueError("fixed EHR/image/report source lineage differs")
        for kind,field in (("ehr","ehr_finding_states"),("xrv","cxr_finding_states"),("chexbert","report_finding_states")):
            if row["states"][kind]!=ref[field][row["finding"]]:
                raise ValueError("raw frozen state changed before scope audit")
        enriched.append({**row,"report_model_id":original["model_id"]})
    items,reader,bundle_sources=load_bundle(args.bundle_run)
    bundle=Path(args.bundle_run).resolve()
    if read_json(bundle/"summary.json")["source_packet_manifest_sha256"]!=sha256_file(packet/"manifest.json"):
        raise ValueError("approved copied reports have a different source packet")
    item_index={row["item_id"]:row["report_sha256"] for row in items}
    if len(items)!=48 or any(item_index[row["item_id"]]!=row["report_sha256"] for row in resolver):
        raise ValueError("copied report inventory differs")
    sources.update({"bundle_"+name:path for name,path in bundle_sources.items()})
    initial={name:sha256_file(source) for name,source in sources.items()}
    # Only approved copied SYNTHETIC text is opened after metadata/pin checks.
    texts={row["report_sha256"]:reader(row["report_sha256"]) for row in items}
    result=build_scope_table(enriched,texts,scope_check)
    if initial!={name:sha256_file(source) for name,source in sources.items()}:
        raise ValueError("immutable source changed during audit")
    return result,sources


def table_csv(records):
    flat=[{key:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value
        for key,value in row.items()} for row in records]
    output=io.StringIO(newline="")
    writer=csv.DictWriter(output,fieldnames=list(flat[0]),lineterminator="\n")
    writer.writeheader();writer.writerows(flat)
    return output.getvalue()


def markdown(summary):
    lines=["# Candidate report-scope availability / 候选报告证据可用性","",
        "Diagnostic only: no new winner, clinical labels, model calls, or regeneration.",
        "Eight-finding denominators are retained; the frozen syntax guard supports only four.","",
        "| Edge | Stage | Inventory | Comparable | Support | Opposition signal | Unknown | Not comparable |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for edge in EDGES:
        for stage in ("raw","scoped"):
            x=summary["relations"][stage][edge]
            lines.append(f"| {edge} | {stage} | {x['inventory_facts']} | {x['comparable_facts']} | {x['support']} | {x['opposition']} | {x['unknown']} | {x['not_comparable']} |")
    lines += ["","Image/EHR rows are deduplicated by CXR byte hash and finding; report edges retain all candidate findings.",
        "Withdrawing an opposition is abstention/lost coverage, NOT a repaired report or improved clinical consistency.",
        "Four reports share a CXR. Cross-path disagreement is a review signal, not independent voting or fault localization.",
        "XRV/CheXbert and syntax coverage are not independent clinical truth. Clinical selection scores stay null.",
        "Human labels have not been supplied. Original score tables and selected triples remain immutable.",""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("packet-run","bundle-run","finding-run","output-root","run-id"):
        parser.add_argument(f"--{name}",required=True)
    args=parser.parse_args();os.umask(0o007);temporary=None
    try:
        temporary,target=new_atomic_run(args.output_root,args.run_id)
        (facts,candidates,groups,summary,assertions),sources=load(args)
        files=[write_private_text(temporary/"fact_scope_table.jsonl","".join(json.dumps(row,sort_keys=True)+"\n" for row in facts)),
            write_private_text(temporary/"candidate_scope_table.csv",table_csv(candidates)),
            write_private_json(temporary/"candidate_scope_table.json",{"records":candidates}),
            write_private_json(temporary/"cross_path_groups.json",{"records":groups}),
            write_private_json(temporary/"report_scope_assertions.json",{"records":assertions}),
            write_private_json(temporary/"summary.json",summary),write_private_text(temporary/"summary.md",markdown(summary))]
        write_private_json(temporary/"manifest.json",{"schema_version":SCHEMA,"run_id":args.run_id,
            "program_sha256":sha256_file(__file__),
            "table_library_sha256":sha256_file(Path(__file__).resolve().parents[1]/"src/tricompose_v12/report_scope_table.py"),
            "source_paths":{name:str(path) for name,path in sources.items()},
            "source_sha256":{name:sha256_file(path) for name,path in sources.items()},
            "model_calls":0,"selection_changed":False,"regeneration_authorized":False,
            "primary_metric_eligible":False,
            "artifacts":{path.name:{"sha256":sha256_file(path)} for path in files}})
        commit_atomic_run(temporary,target)
    except Exception as exc:
        if temporary is not None:discard_atomic_run(temporary)
        print(json.dumps({"status":"failed","error_type":type(exc).__name__}))
        return 1
    print(json.dumps({"status":"completed_candidate_scope_availability","candidate_triples":len(candidates),
        "fact_rows":len(facts),"model_calls":0,"selection_changed":False}))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
