"""Recount committed Push records without a simulator, dataset, or checkpoint."""
from collections import Counter, defaultdict
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'artifacts' / group / 'physical_comparison.json' for group in
             ('chunk_bc', 'act_push', 'act_temporal_ensemble', 'diffusion_push')]
    reports = [json.loads(path.read_text()) for path in paths]
    reference = reports[0]['dataset_indices']
    assert len(reference) == len(set(reference)) == 20
    initial_states = {}
    for path, report in zip(paths, reports):
        assert report['dataset_indices'] == reference
        groups = defaultdict(list)
        for row in report['records']:
            groups[row['backend']].append(row)
        for backend, rows in groups.items():
            assert len(rows) == 20
            assert [r['dataset_index'] for r in rows] == reference
            assert [r['trial'] for r in rows] == list(range(1, 21))
            for row in rows:
                state = (row['seed'], tuple(row['initial_position']))
                previous = initial_states.setdefault(row['dataset_index'], state)
                assert previous == state, (backend, row['dataset_index'])
                if row['success']:
                    assert row['target_satisfied'] and row['displacement'] >= 0.06
            summary = report['summary'][backend] if path == paths[0] else report['summary']
            successes = sum(r['success'] for r in rows)
            failures = dict(Counter(r['failure_reason'] for r in rows if not r['success']))
            assert summary['successes'] == successes
            assert summary['trials'] == len(rows)
            assert summary['failure_reasons'] == failures
            assert summary['target_satisfied'] == sum(r['target_satisfied'] for r in rows)
            print(json.dumps({'source': str(path.relative_to(root)), 'backend': backend,
                              'successes': successes, 'trials': len(rows), 'failures': failures}))
    training = [json.loads((root / 'artifacts' / group / 'training_metrics.json').read_text())
                for group in ('act_push', 'diffusion_push')]
    for report in training:
        assert report['episodes'] == 50 and report['timesteps'] == 32652
        train, validation = set(report['train_episode_ids']), set(report['validation_episode_ids'])
        assert len(train) == 40 and len(validation) == 10 and not train & validation
        assert train | validation == set(range(50))
        print(json.dumps({'matched_states_from_train': len(set(reference) & train),
                          'matched_states_from_validation': len(set(reference) & validation)}))
    assert training[0]['train_episode_ids'] == training[1]['train_episode_ids']
    for report in reports[2:]:
        config = report['configuration']
        assert config['checkpoint_unchanged']
        assert config['checkpoint_sha256_before'] == config['checkpoint_sha256_after']
    print('PASS: saved counts, trial identity, exact initial positions/seeds, and split metadata agree.')
    print('This audits saved evidence; it does not rerun learned policies or verify absent weights.')


if __name__ == '__main__':
    main()
