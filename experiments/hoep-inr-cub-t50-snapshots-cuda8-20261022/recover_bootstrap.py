"""Archive the failed empty bootstrap attempt and reset the queue."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
status = json.loads((root / 'status.json').read_text())
assert status['state'] == 'failed'
for ds, record in status['datasets'].items():
    assert record['returncode'] != 0 and record['phase'] == 'smoke'
    output = Path(record['output'])
    archive = output.with_name(output.name + '.failed_bootstrap')
    assert output.exists() and not archive.exists()
    assert not list(output.iterdir()), f'unexpected training artifacts in {output}'
    output.rename(archive)
    log = Path(record['log'])
    log_archive = log.with_suffix(log.suffix + '.failed_bootstrap')
    assert log.exists() and not log_archive.exists()
    log.rename(log_archive)
(root / 'status.failed_bootstrap.json').write_text(json.dumps(status, indent=2) + '\n')
(root / 'status.json').write_text(json.dumps({'state': 'prepared', 'datasets': {}}, indent=2) + '\n')
print('Archived failed bootstrap; queue reset to prepared')
