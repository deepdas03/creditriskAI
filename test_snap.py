from snapshot_list import list_snapshots, get_snapshot_files
import json

snaps = list_snapshots()
print("found", len(snaps), "snapshots")
if snaps:
    latest = snaps[0]['timestamp']
    print("latest:", latest)
    files_info = get_snapshot_files(latest)
    print(json.dumps(files_info, indent=2))
