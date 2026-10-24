# Release notes

## 0.1.8 — 24 October 2026

Fixes from a review of the reminder and later-message work.

- A reminder asked for after the assistant read a web page, an email, a
  calendar entry, or a list waits for a yes. That text could have written
  the sentence. Otherwise a reminder still saves with no yes.
- A one-time job has to say whether it is a reminder or a one-time report.
  A one-time mail check no longer skips the yes as if it were a reminder.
- A watch that sees unrelated mail stays quiet. A mail check whose inbox
  call failed says the check failed, not that there is nothing to report.
- Asking for a reminder and a question in one message answers both.
- A reminder cannot arrive before the reply that confirms it.
- A database hiccup after a send no longer sends, records, or reports the
  same message twice.
- A reminder may mention scheduling a job interview. A report that asks to
  make another job is still refused.

## 0.1.7 — 22 October 2026

A reminder that starts on a later day does not text on the nights before it.
The first text is that day. The repeat starts after it. A job cannot be a
request to create another job.

A mail check that answers without opening the inbox is a failed check. It
is not reported as an empty inbox. The run looks once more before it says
the check failed.

## 0.1.6 — 18 October 2026

A later message that already reached the other phone still leaves a record
for her assistant. The sender is told it went out only after that record
is stored. A retry writes the record again and does not send the text twice.

A reminder texts the stored sentence on its own.

## 0.1.5 — 18 October 2026

Review of the reminder and later-message work.

- The confirmation includes the name, so the reminder can be cancelled.
- A second message to the other person is stored when the usual name is taken.
- A clock time with a timezone suffix, or a start time that is not text, is refused. It is not sent immediately.
- A reminder saved during a turn that then fails is cancelled.
- A later message that already reached the other phone is not sent again.

## 0.1.4 — 18 October 2026

"Remind Blair in 2 minutes" is a message to her, not a reminder for you.
The assistant calls relay with that delay. An earlier refusal in the chat
was wrong, and the assistant is told so.

## 0.1.3 — 18 October 2026

A message to the other person can go out later.

- Ask to text them at a time, or in a number of minutes or days. Reply yes
  once. That stored text is what they receive, labelled as from you.
- A delay is counted by the server. A named clock time still requires the
  assistant to read the clock in that turn.
- It can be one send, or a repeat until you cancel it.

## 0.1.2 — 18 October 2026

A reminder time comes from the server clock. The assistant cannot save a
clock time it guessed.

- "In two minutes" is a number of seconds. The server adds it to the current
  time. The message you get states that time.
- A named time, such as tomorrow at 8am, is refused until the assistant has
  read the clock in that same turn.
- The 8:53pm check that was stored for 12:44pm the next day is still there
  until you reply yes to the cancel that is waiting.

## 0.1.1 — 18 October 2026

Saying "remind me" saves the reminder. There is no preview and no yes.

- At the time you asked, Signal sends that sentence. "Remind me at 8am to
  put the cleaner in the pool" texts you then: "Don't forget to put the
  cleaner in the pool."
- The first time can be any time in the future. The repeat can be any number
  of days. A minute is the shortest gap.
- A mail digest still waits for a yes, and still writes its own report.

## 0.1.0 — 18 October 2026

A reminder texts the sentence you asked for, at the time you asked.

- One Signal message at a clock time, then the job ends. "Put the cleaner in
  the pool tomorrow at 8am" is that case.
- A daily reminder can start up to a year ahead, then repeat until you cancel.
  The repeat interval is still at most 30 days. The first run is not limited
  to 24 hours.
- A mail digest is unchanged. The assistant still writes that report, and an
  empty one still says there is nothing to report.

## First production release — 16 October 2026

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
- Relay a message. A yes sends the stored draft, now or at the time you
  chose. The recipient's assistant sees who it was from.
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
