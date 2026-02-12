# `data/processed_snapshot/all_I_parts/`

Split gzip parts of the large `all_I=converted_with_original.csv` source table.

## Purpose

GitHub browser uploads are limited to 25 MiB/file. This folder stores the large file as numbered `<25 MiB` chunks so users can reconstruct it locally.

## Rebuild command

```bash
bash scripts/reproducibility/rebuild_all_i_from_parts.sh --work-dir data/processed_snapshot
```

## Output

- `data/processed_snapshot/all_I=converted_with_original.csv`
