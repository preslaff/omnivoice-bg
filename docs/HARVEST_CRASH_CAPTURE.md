# Making a GPU-box freeze diagnosable (crash capture)

**Why this exists.** On **2026-07-28 ~07:07** the box hard-froze during the
stage-2 CPU lattice decode of `harvest_new_sessions.sh`. The postmortem was
inconclusive *because nothing was captured*:

- **Not resource exhaustion** — sar at 07:00 showed **~18.7 GB RAM free**, swap
  idle (0.42 %, flat), **iowait ~0.01 %**, and load *falling* (8→4) as jobs
  finished. Contrast the 2026-07-04 event, which *was* OOM.
- **Not the workload** — stage 2 is single-threaded CPU `gmm-latgen-faster`; the
  only GPU step (stage 3 align) runs `--use-gpu false` and was never reached.
- **Nothing logged** — no OOM-killer line, no NVIDIA/Xid, no thermal/MCE, no
  hung-task, and **no kernel panic in pstore**. The journal simply stops after a
  routine 07:05 cron. An abrupt, total, unlogged hang = a platform-level lockup
  (kernel / driver / hardware). Prime suspect: the new **`7.0.0-28-generic`**
  kernel (first booted Jul 23); the prior `6.17.0-*` kernels have long
  clean-uptime records.

`run_harvest.sh` already covers the *live* trail (hardware watchdog → temps /
power / mem / load every 30 s) and the *durability* (tmux + tee'd log). This doc
is the **one-time host setup** that captures the *cause* of the next freeze —
so if it recurs we get a stack trace or a temperature spike instead of silence.

All commands run on the **GPU box** (not the VPS). Do them once.

---

## 0. First, remove the prime suspect (cheapest test)

Boot the known-good kernel for long batch jobs and see if freezes stop:

```bash
# at boot: hold Shift for GRUB -> Advanced options -> pick 6.17.0-40-generic
uname -r          # confirm 6.17.0-* before launching run_harvest.sh
```

If you want it to *stay* the default while testing, set the GRUB default to the
6.17 entry (`sudo nano /etc/default/grub` → `GRUB_DEFAULT="Advanced...>Ubuntu, with Linux 6.17.0-40-generic"`,
then `sudo update-grub`). Revert once you trust 7.0.

`run_harvest.sh` warns and prompts if you launch on `7.0.0-28-generic`, so you
can't start the harvest on the suspect kernel by accident.

---

## 1. hung-task + panic-on-hang (catch soft hangs before they go total)

A task blocked in the kernel for >120 s often *precedes* a full lockup. Make the
kernel shout about it, and turn a detected hang into a logged panic:

```bash
sudo tee /etc/sysctl.d/99-harvest-watchdog.conf >/dev/null <<'EOF'
# log any task stuck in D-state > 120s (the early-warning we lacked on 07-28)
kernel.hung_task_timeout_secs = 120
# escalate a detected hung task to a real panic (so kdump/pstore records it)
kernel.hung_task_panic = 1
# panic on kernel oops too, and auto-reboot 30s after a panic
kernel.panic_on_oops = 1
kernel.panic = 30
# NMI watchdog for hard lockups (a frozen CPU with irqs off)
kernel.nmi_watchdog = 1
EOF
sudo sysctl --system
```

Trade-off: `hung_task_panic=1` will reboot on a genuine long stall. That's
desirable for an *unattended* batch box — a captured panic + auto-reboot beats a
silent freeze you find hours later. Drop `hung_task_panic`/`panic` to `0` if you
prefer it to hang and wait for you.

---

## 2. pstore-efi (survives a reboot with zero extra RAM)

pstore writes the panic tail to EFI vars / ERST so it's readable *after* the
reboot — exactly what was empty on 07-28. Usually just needs the backend loaded:

```bash
sudo modprobe efi_pstore 2>/dev/null || true
echo efi_pstore | sudo tee /etc/modules-load.d/pstore.conf
mount | grep -q pstore || sudo mount -t pstore pstore /sys/fs/pstore   # normally automounted
ls -la /sys/fs/pstore/     # after a future panic: dmesg-* files land here
```

After any future freeze, first thing to check:

```bash
ls -la /sys/fs/pstore/ ; sudo cat /sys/fs/pstore/dmesg-* 2>/dev/null
journalctl -k -b -1 | tail -60          # tail of the boot that died
```

---

## 3. kdump — the full crash dump (best evidence, ~a few GB reserved)

kdump boots a tiny crash kernel on panic and dumps the full kernel memory image
to `/var/crash`. This is the gold-standard capture; costs some reserved RAM.

```bash
sudo apt install linux-crashdump      # answer "yes" to enable at boot
# it appends crashkernel=... to the kernel cmdline; then reboot once:
sudo systemctl reboot
# verify after reboot:
kdump-config show                     # -> "current state: ready to kdump"
cat /proc/cmdline | grep -o 'crashkernel=[^ ]*'
```

After a future panic the dump is in `/var/crash/<timestamp>/`; open with:

```bash
sudo apt install linux-crashdump crash
crash /usr/lib/debug/boot/vmlinux-$(uname -r) /var/crash/*/dump.*   # 'bt' for the backtrace
```

If a hardware/PCIe fault is suspected instead of a kernel bug, kdump may itself
fail to run — then pstore (§2) and the watchdog hardware log are the fallback.

---

## 4. netconsole (optional — for a truly dead box)

If even pstore comes up empty (hardware so wedged the disk/EFI never gets
written), stream kernel logs over UDP to another machine on the LAN so the last
lines land *off* the box before it dies:

```bash
# on a second machine (receiver): nc -u -l 6666 | tee harvest_netconsole.log
sudo modprobe netconsole netconsole=6666@<gpu-box-ip>/eth0,6666@<receiver-ip>/<receiver-mac>
```

---

## Post-freeze checklist (what to run next time)

1. `ls /sys/fs/pstore/` and `sudo cat /sys/fs/pstore/dmesg-*` — panic tail (§2).
2. `ls /var/crash/` — kdump image (§3); `crash` for the backtrace.
3. `journalctl -k -b -1 | tail -80` — last kernel lines of the dead boot.
4. `tail -40 logs/harvest_hw_*.log` — **the watchdog's last block: temps/power at
   the wall** (added by `run_harvest.sh`, the trail we lacked on 07-28).
5. `sar -r -f /var/log/sysstat/sa$(date +%d) | tail` and `-q`, `-u` — mem / load /
   iowait trajectory (this is what proved 07-28 was *not* OOM).
6. `uname -r` at the time — kernel correlation (6.17 clean vs 7.0 suspect).

The combination of §1–§4 means the **next** freeze produces evidence
(hung-task trace, panic dump, or a temperature spike) instead of the silence we
got on 2026-07-28.
