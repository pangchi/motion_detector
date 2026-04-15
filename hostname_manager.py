#!/usr/bin/env python3
"""
hostname_manager.py

A utility class to manage Raspberry Pi hostname based on
a custom prefix and the last 4 digits of the WLAN MAC address.
Designed to be run on every boot via systemd. Reboots only if
the hostname was changed. Uses a reboot guard to prevent reboot loops.

Usage (standalone):
    sudo python3 hostname_manager.py

Usage (imported):
    from hostname_manager import HostnameManager
    manager = HostnameManager(prefix="rpi-", interface="wlan0")
    manager.apply()
"""

import os
import re
import subprocess
import sys
from pathlib import Path


class HostnameManager:
    """
    Manages the hostname of a Raspberry Pi by combining a custom prefix
    with the last 4 hex digits of the WLAN MAC address.

    Designed to run on every boot via systemd:
      - If hostname matches, does nothing and exits cleanly.
      - If hostname mismatches and no reboot guard exists, updates and reboots once.
      - If hostname mismatches but reboot guard exists, logs a warning and exits
        without rebooting to prevent a reboot loop.

    Args:
        prefix       (str): Custom prefix for the hostname. Defaults to 'rpi-'.
        interface    (str): WLAN network interface name. Defaults to 'wlan0'.
        reboot_delay (int): Seconds to wait before rebooting. Defaults to 5.
        guard_file   (Path): Path to the reboot guard flag file.
    """

    HOSTNAME_FILE = Path("/etc/hostname")
    HOSTS_FILE    = Path("/etc/hosts")
    GUARD_FILE    = Path("/var/lib/hostname-manager/reboot.guard")

    def __init__(
        self,
        prefix: str = "rpi-",
        interface: str = "wlan0",
        reboot_delay: int = 5,
        guard_file: Path = None
    ):
        self.prefix       = prefix
        self.interface    = interface
        self.reboot_delay = reboot_delay
        self.guard_file   = guard_file or self.GUARD_FILE
        self._mac         = None  # Cached MAC address

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_mac(self) -> str:
        """
        Returns the MAC address of the WLAN interface.
        Result is cached after the first call.

        Returns:
            str: MAC address in the format 'xx:xx:xx:xx:xx:xx'.

        Raises:
            FileNotFoundError: If the interface does not exist.
        """
        if self._mac is None:
            mac_path = Path(f"/sys/class/net/{self.interface}/address")
            if not mac_path.exists():
                raise FileNotFoundError(
                    f"Interface '{self.interface}' not found. "
                    f"Check the interface name with 'ip link'."
                )
            self._mac = mac_path.read_text().strip()
        return self._mac

    def get_last4_mac(self) -> str:
        """
        Extracts the last 4 hex characters from the WLAN MAC address.

        Returns:
            str: Last 4 hex characters in uppercase, e.g. '2B3C'.
        """
        mac_clean = self.get_mac().replace(":", "").replace("-", "")
        return mac_clean[-4:].upper()

    def build_hostname(self) -> str:
        """
        Constructs the new hostname from the prefix and last 4 MAC digits.

        Returns:
            str: New hostname in lowercase, e.g. 'rpi-2b3c'.
        """
        return f"{self.prefix}{self.get_last4_mac()}".lower()

    def get_current_hostname(self) -> str:
        """
        Reads the current hostname from /etc/hostname.

        Returns:
            str: The current hostname.
        """
        return self.HOSTNAME_FILE.read_text().strip()

    def is_hostname_correct(self) -> bool:
        """
        Checks whether the current hostname already matches
        the expected hostname built from the MAC address.

        Returns:
            bool: True if hostname matches, False otherwise.
        """
        return self.get_current_hostname() == self.build_hostname()

    def apply(self, dry_run: bool = False) -> str:
        """
        Checks the hostname on every boot and applies a correction
        if there is a mismatch. Reboots automatically if a change
        was made. Uses a guard file to prevent reboot loops.

        Args:
            dry_run (bool): If True, prints what would change without
                            making any modifications. Defaults to False.

        Returns:
            str: The new hostname that was (or would be) applied.

        Raises:
            ValueError:                    If the constructed hostname is invalid.
            PermissionError:               If the script lacks root privileges.
            subprocess.CalledProcessError: If hostnamectl or reboot fails.
        """
        new_hostname = self.build_hostname()
        old_hostname = self.get_current_hostname()

        self._validate_hostname(new_hostname)

        print(f"WLAN interface   : {self.interface}")
        print(f"WLAN MAC address : {self.get_mac()}")
        print(f"Last 4 MAC digits: {self.get_last4_mac()}")
        print(f"Current hostname : {old_hostname}")
        print(f"Expected hostname: {new_hostname}")

        if self.is_hostname_correct():
            print("Hostname is correct. No changes needed.")
            self._clear_guard()
            return new_hostname

        print(f"Hostname mismatch detected: '{old_hostname}' -> '{new_hostname}'")

        # Guard check — if we already rebooted once and hostname is still
        # wrong, something else is overwriting it. Stop rebooting and log.
        if self._guard_exists():
            print(
                "WARNING: Reboot guard is set but hostname is still mismatched. "
                "Another process may be overwriting /etc/hostname. "
                "Skipping reboot to prevent a reboot loop. "
                "Please investigate manually."
            )
            return new_hostname

        if dry_run:
            print(f"[Dry run] Hostname would be updated to '{new_hostname}'.")
            print(f"[Dry run] System would reboot in {self.reboot_delay} second(s).")
            return new_hostname

        self._check_root()
        self._write_hostname(new_hostname)
        self._write_guard()

        print("Hostname updated successfully.")
        self._reboot()

        return new_hostname

    # ------------------------------------------------------------------
    # Guard file helpers
    # ------------------------------------------------------------------

    def _guard_exists(self) -> bool:
        """Returns True if the reboot guard file exists."""
        return self.guard_file.exists()

    def _write_guard(self):
        """
        Creates the reboot guard file to signal that a reboot
        has already been triggered for a hostname change.
        """
        self.guard_file.parent.mkdir(parents=True, exist_ok=True)
        self.guard_file.write_text(
            f"Rebooted to apply hostname: {self.build_hostname()}\n"
        )
        print(f"Reboot guard written to {self.guard_file}")

    def _clear_guard(self):
        """
        Removes the reboot guard file once the hostname is confirmed
        correct, so future hostname changes can trigger a reboot again.
        """
        if self._guard_exists():
            self.guard_file.unlink()
            print(f"Reboot guard cleared from {self.guard_file}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_hostname(self, hostname: str):
        """Validates the hostname against RFC 1123 rules."""
        pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?$'
        if not re.match(pattern, hostname):
            raise ValueError(
                f"'{hostname}' is not a valid hostname. "
                f"Only alphanumeric characters and hyphens are allowed."
            )

    def _check_root(self):
        """Raises PermissionError if the script is not run as root."""
        if os.geteuid() != 0:
            raise PermissionError(
                "Root privileges are required to modify hostname files. "
                "Please run with sudo."
            )

    def _write_hostname(self, new_hostname: str):
        try:
            # 1. Update the persistent cloud-init config to prevent reverts
            cloud_config = "/etc/cloud/cloud.cfg"
            if os.path.exists(cloud_config):
                subprocess.run(['sudo', 'sed', '-i', 's/preserve_hostname: false/preserve_hostname: true/', cloud_config])
            
            # 2. Set the hostname using hostnamectl
            subprocess.run(['sudo', 'hostnamectl', 'set-hostname', new_hostname], check=True)
            
            # 3. Update /etc/hosts to prevent sudo resolution errors
            # Replaces the old hostname associated with 127.0.1.1
            subprocess.run(['sudo', 'sed', '-i', f's/127.0.1.1.*/127.0.1.1\t{new_hostname}/g', '/etc/hosts'], check=True)
            
            print(f"Hostname successfully changed to '{new_hostname}'.")
            print("A reboot is required for all changes to take effect.")
            
        except subprocess.CalledProcessError as e:
            print(f"Error changing hostname: {e}")

    def _reboot(self):
        """Schedules a system reboot after reboot_delay seconds."""
        print(f"System will reboot in {self.reboot_delay} second(s) to apply changes...")
        if self.reboot_delay >= 60:
            subprocess.run(
                ["shutdown", "-r", f"+{self.reboot_delay // 60}"],
                check=True
            )
        else:
            self._reboot_with_sleep()

    def _reboot_with_sleep(self):
        """Reboots after a short delay using sleep for sub-minute delays."""
        subprocess.Popen(
            f"sleep {self.reboot_delay} && reboot",
            shell=True
        )
        print(f"Reboot scheduled in {self.reboot_delay} second(s). Exiting...")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HostnameManager(prefix={self.prefix!r}, "
            f"interface={self.interface!r}, "
            f"reboot_delay={self.reboot_delay!r})"
        )

    def __str__(self) -> str:
        try:
            return (
                f"HostnameManager | interface: {self.interface} | "
                f"MAC: {self.get_mac()} | "
                f"Proposed hostname: {self.build_hostname()}"
            )
        except FileNotFoundError as e:
            return f"HostnameManager | Error: {e}"


# ----------------------------------------------------------------------
# Standalone entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    manager = HostnameManager(prefix="motiondetector-", interface="wlan0", reboot_delay=15)
    try:
        manager.apply()
    except (FileNotFoundError, ValueError, PermissionError) as e:
        print(f"Error: {e}")
        sys.exit(1)
