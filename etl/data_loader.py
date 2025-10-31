import json
import re
from datetime import datetime
from matplotlib import pyplot as plt
from etl.zip_inspector import decode_key, try_decode_value


def load_db0(selected_manifest, selected_backups):
    selected_db0_file_name = None
    if selected_manifest:
        files_map = selected_manifest.get('files', {}) or {}
        selected_db0_file_name = files_map.get('0')

    selected_db0_backup = None
    if selected_backups and selected_db0_file_name:
        selected_db0_backup = selected_backups.get(selected_db0_file_name)

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

    return {item['key']: item for item in selected_db0_values}


def render_db0_sample_timeseries(entry_map):
    collected = []
    for key, entry in entry_map.items():
        if not (key.startswith('sample_') and not key.endswith('_meta')):
            continue
        series = _extract_sample_series(entry)
        if not series:
            continue
        meta_entry = entry_map.get(f"{key}_meta")
        timestamp = _extract_sample_timestamp(meta_entry) if isinstance(meta_entry, dict) else None
        collected.append((timestamp, key, series))
    if not collected:
        raise ValueError('No sample time series available.')

    def _sort_key(item):
        ts, key, _ = item
        if ts:
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                pass
        return key
    collected.sort(key=_sort_key)
    plt.figure(figsize=(12, 4))
    for timestamp, key, series in collected:
        xs = list(range(len(series)))
        label = f"{key} ({timestamp})" if timestamp else key
        plt.plot(xs, series, linewidth=1, label=label)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Sample time series overview')
    if len(collected) <= 12:
        legend_cols = 1 if len(collected) <= 6 else 2
        plt.legend(fontsize=6, ncol=legend_cols, frameon=False)
    elif len(collected) <= 24:
        plt.legend(fontsize=6, ncol=3, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()
    cols = 2
    rows = (len(collected) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3), squeeze=False)
    flat_axes = axes.flatten()
    for ax in flat_axes[len(collected):]:
        ax.axis('off')
    for ax, (timestamp, key, series) in zip(flat_axes, collected):
        ax.plot(range(len(series)), series, marker='o', linewidth=1)
        title = f"{key} ({timestamp})" if timestamp else key
        ax.set_title(title, fontsize=8)
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
    plt.tight_layout()
    plt.show()


def _extract_sample_series(entry):
    if not isinstance(entry, dict):
        return []
    value_json = entry.get('value_json')
    series = _coerce_numeric_series(value_json)
    if series:
        return series
    value_text = entry.get('value_text')
    if isinstance(value_text, str):
        series = _coerce_numeric_series(value_text)
        if series:
            return series
    value_bytes = entry.get('value_bytes')
    if isinstance(value_bytes, (bytes, bytearray)):
        try:
            decoded = value_bytes.decode('utf-8', errors='replace')
        except Exception:
            decoded = ''
        if decoded:
            series = _coerce_numeric_series(decoded)
            if series:
                return series
    details = entry.get('details')
    if isinstance(details, dict):
        decoded_bytes = details.get('decoded_bytes')
        if isinstance(decoded_bytes, (bytes, bytearray)):
            try:
                decoded = decoded_bytes.decode('utf-8', errors='replace')
            except Exception:
                decoded = ''
            if decoded:
                series = _coerce_numeric_series(decoded)
                if series:
                    return series
    return []


def _extract_sample_timestamp(meta_entry):
    if not isinstance(meta_entry, dict):
        return None
    value_json = meta_entry.get('value_json')
    if isinstance(value_json, dict):
        for key in ('timestamp', 'created_at', 'created'):
            ts = value_json.get(key)
            if isinstance(ts, str) and ts:
                return ts
    value_text = meta_entry.get('value_text') if isinstance(meta_entry, dict) else None
    if isinstance(value_text, str):
        trimmed = value_text.strip()
        if trimmed.startswith('{') and trimmed.endswith('}'):
            try:
                parsed = json.loads(trimmed)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                for key in ('timestamp', 'created_at', 'created'):
                    ts = parsed.get(key)
                    if isinstance(ts, str) and ts:
                        return ts
        else:
            return trimmed or None
    return None


def _coerce_numeric_series(data):
    if data is None:
        return []
    if isinstance(data, bool):
        return []
    if isinstance(data, (int, float)):
        return [float(data)]
    if isinstance(data, str):
        trimmed = data.strip()
        if not trimmed:
            return []
        try:
            return [float(trimmed)]
        except ValueError:
            try:
                parsed = json.loads(trimmed)
            except json.JSONDecodeError:
                tokens = []
                for token in trimmed.replace(',', ' ').split():
                    try:
                        tokens.append(float(token))
                    except ValueError:
                        continue
                return tokens
            else:
                return _coerce_numeric_series(parsed)
    if isinstance(data, (list, tuple)):
        collected = []
        for item in data:
            if isinstance(item, dict):
                handled = False
                for candidate in ('value', 'y', 'val', 'score'):
                    if candidate in item:
                        nested = _coerce_numeric_series(item[candidate])
                        if nested:
                            collected.extend(nested)
                            handled = True
                        break
                if handled:
                    continue
                nested = _coerce_numeric_series(list(item.values()))
                if nested:
                    collected.extend(nested)
            else:
                nested = _coerce_numeric_series(item)
                if nested:
                    collected.extend(nested)
        return collected
    if isinstance(data, dict):
        for candidate in (
                'series',
                'values',
                'data',
                'points',
                'samples',
                'sample',
                'payload',
                'entries',
                'items',
                'measurements',
                'sample_dict',
        ):
            if candidate in data:
                nested = _coerce_numeric_series(data[candidate])
                if nested:
                    return nested
        numeric_items = []
        for idx, (key, value) in enumerate(data.items()):
            floats = _coerce_numeric_series(value)
            if not floats:
                continue
            if len(floats) == 1:
                numeric_items.append(((idx, 0), key, floats[0]))
            else:
                for offset, val in enumerate(floats):
                    numeric_items.append(((idx, offset), f"{key}[{offset}]", val))
        if not numeric_items:
            return []
        def _dict_sort_key(order, raw_key):
            order_token = tuple(order)
            if isinstance(raw_key, (int, float)):
                return (0, float(raw_key), order_token)
            if isinstance(raw_key, str):
                stripped = raw_key.strip()
                try:
                    return (0, float(stripped), order_token)
                except ValueError:
                    numbers = re.findall(r'-?\d+(?:\.\d+)?', stripped)
                    if numbers:
                        return (1, tuple(float(num) for num in numbers), order_token)
                    return (2, stripped.lower(), order_token)
            return (3, str(raw_key), order_token)
        numeric_items.sort(key=lambda item: _dict_sort_key(item[0], item[1]))
        return [value for _, _, value in numeric_items]
    return []
