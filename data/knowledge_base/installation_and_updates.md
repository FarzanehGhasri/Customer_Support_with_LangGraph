# Installing and updating the app

**Keywords:** install, installation, update, upgrade, download, system requirements, uninstall

## System requirements

* **Windows:** 10 (build 19041) or newer, 4 GB RAM
* **macOS:** 12 Monterey or newer, Apple Silicon or Intel
* **Linux:** Ubuntu 22.04+ / Fedora 38+, `glibc` 2.35 or newer
* **Mobile:** iOS 16+, Android 10+

## Updating

The app checks for updates on launch and installs them in the background.
To update manually, use **Help -> Check for updates**. Enterprise installations
are updated by the administrator through the MSI/PKG package.

## Installation fails

* *Windows error 1603*: run the installer as administrator and make sure the
  previous version is fully uninstalled.
* *macOS "app is damaged"*: the download was incomplete -- download again from
  the official site; do not unblock the quarantine flag manually.

## Uninstalling

Remove the app through your operating system, then delete the data directory if
you want a clean state: `%APPDATA%\ExampleApp` (Windows),
`~/Library/Application Support/ExampleApp` (macOS), `~/.config/exampleapp` (Linux).
