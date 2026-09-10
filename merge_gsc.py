import json
from pathlib import Path

# Find latest result files
results_dir = Path('.')
result_files = sorted(results_dir.glob('result_lingbot-v2_*.json'))

if len(result_files) < 2:
    print("ERROR: Need at least 2 result files (existing + GSC)")
    exit(1)

existing_file = result_files[-2]  # 2nd newest (before GSC run)
gsc_file = result_files[-1]       # newest (GSC run)

print(f"Merging {gsc_file.name} into {existing_file.name}")

with open(existing_file) as f:
    existing = json.load(f)

with open(gsc_file) as f:
    gsc_results = json.load(f)

# Merge GSC by path
path_to_gsc = {e['path']: e.get('gsc') for e in gsc_results['data']}

for entry in existing['data']:
    if entry['path'] in path_to_gsc:
        entry['gsc'] = path_to_gsc[entry['path']]

# Save merged
output = existing_file.with_stem(existing_file.stem + '_with_gsc')
with open(output, 'w') as f:
    json.dump(existing, f)

print(f"✓ Merged to {output.name}")
print(f"  Run: scores.bat lingbot-v2")
