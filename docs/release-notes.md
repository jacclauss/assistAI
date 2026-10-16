# Release notes

First production release. 16 October 2026.

AssistAI runs on a Raspberry Pi 5. Two people text one bot number. Each reaches
their own assistant. An organizer process holds the household lists, and
nobody texts it.

The Mac stack is stopped. The Pi survived a reboot the same day, and both
phones got an answer afterward. A week unattended is the remaining
observation. It started with this release.

## What a person can do

- Ask in Signal. The last 30 turns, and anything from the last 14 days, are
  what the assistant still knows after a restart.
- Read today's shared iCloud calendar. Adding an event waits for a yes.
- Read Gmail, file it, and save a draft. The assistant cannot send mail and
  cannot permanently delete it.
- Keep shared lists and private lists. A private list stays with its owner
  until a separate yes publishes it. The other person gets the same answer as
  for a list that was never created.
- Relay a message. A yes sends the stored draft. The recipient's assistant
  sees who it was from.
- Schedule a job or a watch from Signal, and cancel it there. A schedule
  always texts. A watch stays quiet unless it finds something, fails, or
  expires.
- Look something up. The answer cites its sources. Page text is untrusted, and
  a later yes still has to confirm a send, a filing, or a calendar write.
- Hear from the operator's phone when signal-cli is behind its pin. Nothing
  is installed automatically.

A number that is not allowed gets silence.

## This deploy

- The gateway image is `linux/arm64`, loaded on the Pi. signal-cli `0.100`
  and SearXNG are pulled there. No ports are published.
- The bot number already in use was linked as a device named `assistai`.
  The Mac linked device is the one to remove.
- The Fireworks key on the Pi is a new key. The calendar password is an
  app-specific password in a mode-600 file outside the repo. SearXNG has its
  own secret. Gmail sign-ins for both people are in the state volume.
- The Google OAuth app is in production, so those sign-ins are no longer on
  the 7-day testing clock. The app is unverified. Each person continues past
  Google's warning once.
- Fireworks retired `deepseek-v4-flash-0731`. The quarantine model is now
  `deepseek-v4p1-flash`. The primary model is still `minimax-m3`. Both
  answered at startup after the reboot.
- Later image installs are in [pi-deploy.md](pi-deploy.md). GitHub does not
  deploy to the Pi. SSH is still on until it is turned off from that runbook.

## Not in this release

- Diet and fitness specialists. That work is a draft and is not running.
- Signal attachment bytes. A file is noted. The sender pastes a link to pass
  it on.
- Automatic updates. The watcher texts, and the operator loads a new image.
- The Docker memory caps in the compose file. This Pi's kernel discarded
  them. The containers still restart with the host.
