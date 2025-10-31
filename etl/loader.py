import json
from pathlib import Path
import zipfile, base64

from etl.data_loader import load_db0, render_db0_sample_timeseries
from etl.logs_loader import render_worker_report

from etl.zip_inspector import scan_and_load, parse_zip_metadata, collect_archive_data, decode_key, try_decode_value, \
    summarise_entry_generic, DB_LABELS, _fetch_backup_metadata, summarise_entry

DB_NAMES = {
    0: "data",
    1: "candidates",
    2: "reasons",
    3: "non_reasons",
    4: "candidate_anti_reasons",
    5: "anti_reasons",
    6: "good_profiles",
    7: "bad_profiles",
    8: "preferred_reasons",
    9: "anti_reason_profiles",
}


def etl(zip_paths, results_dir):
    selected_zip_name = scan_and_load(zip_paths, results_dir)
    archives_data = [collect_archive_data(path) for path in zip_paths]
    manifests_by_archive = {item['zip_name']: item['manifest'] for item in archives_data}
    manifest_prefix_by_archive = {item['zip_name']: item.get('manifest_prefix', '') for item in archives_data}
    backups_by_archive = {item['zip_name']: item['backups'] for item in archives_data}
    selected_archive_data = next((item for item in archives_data if item['zip_name'] == selected_zip_name), None)
    selected_manifest = manifests_by_archive.get(selected_zip_name)
    selected_manifest_prefix = manifest_prefix_by_archive.get(selected_zip_name, '')
    selected_backups = backups_by_archive.get(selected_zip_name)


    workers_table, workers_table2, workers_table3, plots = render_worker_report(selected_zip_name, selected_manifest, selected_backups, selected_archive_data, selected_manifest_prefix)

    db = { DB_NAMES.get(0) : load_db0(selected_manifest, selected_backups)}

    render_db0_sample_timeseries(db[DB_NAMES.get(0)])

    if selected_manifest and selected_archive_data and selected_manifest_prefix is not None:
        files_map = (selected_manifest.get('files') or {})
        selected_zip_path = Path(selected_archive_data['zip_path'])
        with zipfile.ZipFile(selected_zip_path) as z:
            for db_index in range(1, 10):
                file_name = files_map.get(str(db_index))
                if not file_name:
                    continue
                member_name = f"{selected_manifest_prefix}{file_name}"
                try:
                    raw_text = z.read(member_name).decode("utf-8", errors="replace")
                    obj = json.loads(raw_text)
                except Exception:
                    db[db_index] = {'file_name': file_name, 'entries': []}
                    continue

                entries_map = {}
                for entry in obj.get('entries') or []:
                    try:
                        key_bytes = decode_key(entry)
                        key_text = key_bytes.decode('utf-8', errors='replace')
                    except Exception as exc:
                        key_text = f'<unable to decode key: {exc}>'

                    preview, details = try_decode_value(entry)
                    value_bytes = details.get('decoded_bytes') if isinstance(details, dict) else None

                    if not isinstance(value_bytes, (bytes, bytearray)):
                        v = entry.get('value') or {}
                        if isinstance(v, dict) and isinstance(v.get('data'), str):
                            try:
                                value_bytes = base64.b64decode(v['data'], validate=False)
                            except Exception:
                                value_bytes = None

                    if isinstance(value_bytes, (bytes, bytearray)):
                        plain_value = value_bytes.decode('utf-8', errors='replace')
                    else:
                        plain_value = str(preview)

                    try:
                        entries_map[key_text] = json.loads(plain_value)
                    except Exception:
                        entries_map[key_text] = plain_value

                name = DB_NAMES.get(db_index, f"db_{db_index}")
                db[name] = entries_map

    return db, workers_table, workers_table2, workers_table3, plots



