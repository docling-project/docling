"""Validate saved visual annotations and their PDF identities; stdlib only."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data = root / 'tests/data/groundtruth/acroform_keying'
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=Path(manifest['fixture_root_hint']))
    args = parser.parse_args()
    pages = {}
    for fixture in manifest['fixtures']:
        pdf = args.fixtures / fixture['filename']
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() == fixture['sha256'], pdf
        for page in fixture['pages']:
            pages[fixture['id'], page['number']] = page
    widgets = {}
    for line in (data / 'widgets.jsonl').read_text(encoding='utf-8').splitlines():
        widget = json.loads(line)
        key = (widget['fixture'], widget['page'], widget['widget_index'])
        assert key not in widgets, key
        assert key[2] in pages[key[:2]]['native_widget_order'], key
        widgets[key] = widget
    for key, page in pages.items():
        assert [w[2] for w in widgets if w[:2] == key] == page['native_widget_order'], key
    annotations = [json.loads(line) for line in (data / 'annotations.jsonl').read_text(encoding='utf-8').splitlines()]
    ids = set()
    table_widgets = 0
    for ann in annotations:
        assert ann['id'] not in ids, ann['id']
        ids.add(ann['id'])
        key = (ann['fixture'], ann['page'])
        page = pages[key]
        kind = ann['kind']
        assert kind in {'label_link', 'group', 'table_region', 'review_note'}, kind
        indices = [ann['widget_index']] if kind == 'label_link' else ann.get('widget_indices', [])
        if kind == 'table_region':
            assert ann['cell_mapping_status'] in {'annotated', 'not_annotated'}
            assert (ann['cell_mapping_status'] == 'annotated') == bool(ann['cells'])
            cell_ids = set()
            for cell in ann['cells']:
                assert 0 <= cell['row'] < len(ann['rows'])
                assert 0 <= cell['column'] < len(ann['columns'])
                cell_id = (cell['row'], cell['column'])
                assert cell_id not in cell_ids
                cell_ids.add(cell_id)
                indices.extend(cell['widget_indices'])
                table_widgets += len(cell['widget_indices'])
        assert len(indices) == len(set(indices)), ann['id']
        for index in indices:
            assert (*key, index) in widgets, (ann['id'], index)
        if kind == 'group':
            assert indices == [i for i in page['native_widget_order'] if i in indices], ann['id']
        bbox = ann.get('bbox') or ann.get('expected_label', {}).get('bbox')
        if bbox is not None:
            l, t, r, b = bbox
            assert 0 <= l < r <= page['width'] and 0 <= t < b <= page['height'], ann['id']
    assert sum(a.get('diagnostic_102_sample', False) for a in annotations) == 102
    print(f"Verified {len(manifest['fixtures'])} PDF hashes, {len(pages)} pages, {len(widgets)} native identities.")
    print(dict(Counter(a['kind'] for a in annotations)))
    print(f'Explicit table-cell membership: {table_widgets} widgets.')


if __name__ == '__main__':
    main()
