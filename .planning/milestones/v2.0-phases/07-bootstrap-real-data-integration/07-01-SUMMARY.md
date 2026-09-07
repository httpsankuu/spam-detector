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
