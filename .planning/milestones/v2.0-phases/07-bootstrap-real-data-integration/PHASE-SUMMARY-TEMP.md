# Plan 07-01 Summary: Fix SMSDataLoader fallback path

**Phase:** 07 - Bootstrap & Real Data Integration
**Plan:** 07-01

## Work Completed
- Audited sms_loader.py and determined the old logic attempted to return a non-existent path on failure.
- Added BUNDLED_SAMPLE_PATH and UCI_BACKUP_URL constants to the SMSDataLoader class.
- Replaced the single xcept block in download_uci_dataset() with a robust three-tier fallback mechanism:
  1. Primary: Download zip from UCI repository.
  2. Backup: Download tsv file from the backup mirror on GitHub.
  3. Fallback: Return the local bundled sample_sms.csv.
- Added a load_any_sms_file() dispatcher that handles TSV, CSV, and headerless TSV formats transparently.

## Verification
- Confirmed load_any_sms_file can read the bundled sample without errors.
- Confirmed download_uci_dataset returns an existing path even when network mock fails.
- All tasks committed atomically.

This plan is successfully completed and requirements DATA-01 and REPRO-02 are addressed for this layer.
# Plan 07-02 Summary: Create scripts/bootstrap.py

**Phase:** 07 - Bootstrap & Real Data Integration
**Plan:** 07-02

## Work Completed
- Created scripts/__init__.py.
- Created scripts/bootstrap.py which ties together:
  - Environment verification (setup_env.py)
  - Dataset download (SMSDataLoader)
  - Data loading and splitting (split_dataset)
  - Model training and persisting (	rainer.py)
  - Model evaluation on test data (metrics.py and plots.py)
- Verified all models train correctly on the real UCI dataset without crashing.
- Verified test suite passes using the updated data structures.

## Verification
- Executed ootstrap.py successfully end-to-end. Benchmark JSON and figures were saved successfully.
- Pytest test suite executed successfully (	est_pipeline.py).
- Fixes REPRO-01 (pipeline script) and REPRO-03 (tests pass).

All requirements are fulfilled.
