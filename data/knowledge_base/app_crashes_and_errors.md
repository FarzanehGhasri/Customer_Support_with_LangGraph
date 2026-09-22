# App crashes, freezes and error codes

**Keywords:** crash, crashing, freeze, hangs, error, bug, not responding, white screen

## First steps for any crash

1. Update to the latest version (**Help -> Check for updates**).
2. Restart the app, then restart the device.
3. Clear the local cache: **Settings -> Advanced -> Clear cache**. This does not
   delete your data, only temporary files.
4. If the crash persists, export the log from **Help -> Export diagnostics** and
   attach it to your support ticket.

## Known error codes

| Code | Meaning | Fix |
| ---- | ------- | --- |
| E-101 | Cannot reach the sync server | Check your network, then retry. Usually transient. |
| E-204 | Local database is corrupted | Run **Settings -> Advanced -> Repair database**. |
| E-403 | Session expired | Sign out and sign in again. |
| E-500 | Server-side failure | Nothing to do locally; check status.example.com. |

## White screen on launch (desktop)

Caused by a stale GPU cache. Launch the app once with the `--disable-gpu` flag,
then clear the cache as described above and launch normally.
