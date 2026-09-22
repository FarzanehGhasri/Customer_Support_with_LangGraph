# Sync problems and offline mode

**Keywords:** sync, not syncing, offline, out of date, conflict, missing data

## Data is not syncing

1. Confirm the sync indicator in the status bar is green. Grey means offline.
2. Force a sync with **File -> Sync now** (or pull down on mobile).
3. Sync pauses automatically on metered connections. Allow it under
   **Settings -> Sync -> Sync on metered networks**.
4. A single item larger than 50 MB is skipped by the sync engine and reported in
   the sync log.

## Resolving a sync conflict

When the same item is edited on two devices, we keep both copies and mark the
newer one "(conflict)". Open **Settings -> Sync -> Conflicts** to pick the
version to keep; the other is kept in the trash for 30 days.

## Working offline

Everything you opened in the last 30 days is cached and editable offline.
Changes queue locally and upload on the next connection.
