from __future__ import annotations
import argparse
from genos_tools.config import load_profile_config
from genos_tools.commands import export_chunks

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="genos-tools", description="GenOS utilities")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export-chunks", help="Export Weaviate chunks by vdb_id")
    e.add_argument("--profile", default="bok", help="profile name in config.toml")
    e.add_argument("--config", help="path to config.toml (optional)")
    e.add_argument("--vdb-id", type=int, required=True, help="target vdb_id")
    e.add_argument("--output-dir", help="override output dir")
    return p

def main():
    args = _parser().parse_args()
    print(args)
    cfg = load_profile_config(args.profile, args.config)
    print("Test")
    if args.cmd == "export-chunks":
        export_chunks.main(cfg, args.vdb_id, args.output_dir)

if __name__ == "__main__":
    main()
