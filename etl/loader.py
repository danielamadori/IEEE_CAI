import json
from pathlib import Path

from etl.logs_loader import summarise_db10_workers

from etl.zip_inspector import scan_and_load, parse_zip_metadata, collect_archive_data, decode_key, try_decode_value, \
    summarise_entry_generic, DB_LABELS, _fetch_backup_metadata, summarise_entry


def etl(zip_paths, results_dir):
    selected_zip_name = scan_and_load(zip_paths, results_dir)
    archives_metadata = [parse_zip_metadata(path) for path in zip_paths]
    archives_data = [collect_archive_data(path) for path in zip_paths]
    manifests_by_archive = {item['zip_name']: item['manifest'] for item in archives_data}
    manifest_prefix_by_archive = {item['zip_name']: item.get('manifest_prefix', '') for item in archives_data}
    backups_by_archive = {item['zip_name']: item['backups'] for item in archives_data}
    selected_archive_data = next((item for item in archives_data if item['zip_name'] == selected_zip_name), None)
    selected_manifest = manifests_by_archive.get(selected_zip_name)
    selected_manifest_prefix = manifest_prefix_by_archive.get(selected_zip_name, '')
    selected_backups = backups_by_archive.get(selected_zip_name)

    summarise_db10_workers(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix)

    selected_db0_file_name = None
    if selected_manifest:
        files_map = selected_manifest.get('files', {}) or {}
        selected_db0_file_name = files_map.get('0')

    selected_db0_backup = None
    if selected_backups and selected_db0_file_name:
        selected_db0_backup = selected_backups.get(selected_db0_file_name)

    selected_db0_entries = []
    selected_db0_values = []
    if selected_db0_backup:
        selected_db0_entries = selected_db0_backup.get('entries') or []
        for entry in selected_db0_entries:
            try:
                key_bytes = decode_key(entry)
                key_text = key_bytes.decode('utf-8', errors='replace')
            except Exception as exc:
                key_text = f'<unable to decode key: {exc}>'
            preview, details = try_decode_value(entry)
            value_bytes = details.get('decoded_bytes') if isinstance(details, dict) else None
            if isinstance(value_bytes, (bytes, bytearray)):
                value_text = value_bytes.decode('utf-8', errors='replace')
            else:
                value_text = str(preview)
            value_json = None
            if isinstance(value_text, str):
                try:
                    value_json = json.loads(value_text)
                except Exception:
                    value_json = None
            selected_db0_values.append({
                'key': key_text,
                'type': entry.get('type'),
                'ttl_ms': entry.get('pttl'),
                'value_text': value_text,
                'value_bytes': value_bytes,
                'value_json': value_json,
                'details': details,
            })

    selected_db0_values_by_key = {item['key']: item for item in selected_db0_values}

    db1_entries = []
    if selected_manifest and selected_backups is not None:
        files_map = selected_manifest.get('files', {}) or {}
        db1_file = files_map.get('1')
        if db1_file:
            data = selected_backups.get(db1_file) if isinstance(selected_backups, dict) else None
            if isinstance(data, dict):
                db1_entries = data.get('entries') or []

    db1_entries_summary = []
    for entry in db1_entries:
        try:
            key_bytes = decode_key(entry)
            key_text = key_bytes.decode('utf-8', errors='replace')
        except Exception as exc:
            key_text = f'<unable to decode key: {exc}>'
        preview, details = try_decode_value(entry)
        value_bytes = details.get('decoded_bytes') if isinstance(details, dict) else None
        if isinstance(value_bytes, (bytes, bytearray)):
            value_text = value_bytes.decode('utf-8', errors='replace')
        else:
            value_text = str(preview)
        value_json = None
        if isinstance(value_text, str):
            try:
                value_json = json.loads(value_text)
            except Exception:
                value_json = None
        headline, extra = summarise_entry_generic(key_text, value_json, value_text)
        db1_entries_summary.append({
            'key': key_text,
            'type': entry.get('type'),
            'ttl_ms': entry.get('pttl'),
            'headline': headline,
            'details': extra,
        })

    other_db_summaries = []
    if selected_manifest and selected_backups is not None:
        files_map = selected_manifest.get('files', {}) or {}
        selected_zip_path = Path(selected_archive_data['zip_path']) if selected_archive_data else None
        for db_index in range(1, 10):
            file_name = files_map.get(str(db_index))
            if not file_name or selected_zip_path is None:
                continue
            label = DB_LABELS.get(db_index, 'Unknown')
            data = selected_backups.get(file_name) if isinstance(selected_backups, dict) else None
            metadata = {}
            if isinstance(data, dict):
                metadata = data.get('metadata') or {}
            if not metadata:
                member_name = f"{selected_manifest_prefix}{file_name}"
                metadata = _fetch_backup_metadata(selected_zip_path, member_name)
            key_count = metadata.get('key_count') if isinstance(metadata, dict) else None
            type_summary = metadata.get('type_summary') if isinstance(metadata, dict) else None
            other_db_summaries.append({
                'db_index': db_index,
                'label': label,
                'file_name': file_name,
                'key_count': key_count,
                'type_summary': type_summary if isinstance(type_summary, dict) else None,
            })

    selected_db0_summary = []
    for item in selected_db0_values:
        headline, extra = summarise_entry(item['key'], item['value_json'], item['value_text'], skip_sample_keys=True)
        if headline is None and extra is None:
            continue
        selected_db0_summary.append({
            'key': item['key'],
            'type': item['type'],
            'ttl_ms': item['ttl_ms'],
            'headline': headline,
            'details': extra,
        })

    if selected_db0_summary:
        print(f"DB 0 entries for {selected_zip_name}:")
        for entry in selected_db0_summary:
            ttl = entry['ttl_ms'] if isinstance(entry['ttl_ms'], int) else 'persistent'
            print(f"{entry['key']} (type={entry['type']}, ttl={ttl})")
            print(f"{entry['headline']}")
            for detail in entry['details']:
                print(f"    {detail}")
    else:
        print('No DB 0 data available for the current selection.')



    return selected_db0_values_by_key, selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix


