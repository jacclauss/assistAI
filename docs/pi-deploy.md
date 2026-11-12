# Pi deployment

The Pi is the always-on host. This Mac builds the arm64 image and copies it
over SSH. GitHub holds the source and can run tests. It does not install
anything on the Pi. The update watcher texts the operator when signal-cli is
behind its pin, and it leaves the install unchanged.

Turn SSH off only after the checks at the end of this document pass. The next
image is installed by turning SSH back on from a keyboard and monitor.

The account on the Pi is `aissist-admin`. The hostname is `raspberrypi.local`.
The LAN address changes. Docker Engine is installed, and `aissist-admin` is in
the `docker` group. This Mac's SSH key is already authorized.

## Production credentials

These stay off the Pi image, off git, and out of chat.

| What | Where it lives on the Pi | Cut over from development |
| --- | --- | --- |
| Fireworks key | `.env` `FIREWORKS_API_KEY` | Create a new key in the Fireworks console. Write it only into the Pi `.env`. After a text succeeds, revoke the key that was used on the Mac. |
| Signal account | `.env` `ASSISTAI_SIGNAL_ACCOUNT` and the `signal-cli-config` volume | Keep the bot number already registered. The Pi links as a device. The Mac device is then removed. |
| Operators and roster | `.env` `ASSISTAI_SIGNAL_ALLOW_FROM` and `config/assistai.toml` | Copy the live roster. The example file has placeholder numbers. |
| SearXNG secret | `.env` `SEARXNG_SECRET` | Generate a new value. Compose's default is for a machine with no published ports during development. |
| Household time zone | `.env` `ASSISTAI_TIMEZONE`, for example `America/Chicago` | Required. The gateway refuses to start Signal without it. Copy the value from the Mac. |
| Calendar | `.env` username and calendar name. Password in `/var/lib/assistai/caldav-password` | The Mac reads the Keychain. The Pi reads only that file. |
| Gmail client | `.env` `ASSISTAI_GMAIL_CLIENT_ID` and `ASSISTAI_GMAIL_CLIENT_SECRET` | Same OAuth client. The consent screen has to leave Testing. |
| Gmail sign-in | `assistai_gateway-state` volume, per agent | Copy the volume, or sign in again on the Pi. A Testing app's refresh token expires after 7 days. |

`ASSISTAI_LOG_CONSOLE` stays false. `ASSISTAI_SIGNAL_DM_POLICY` stays
`allowlist`. The calendar password is not a line in `.env`. Do not start the
Pi with `docker/compose.dev.yml`.

### Gmail

In Google Cloud, move the OAuth consent screen from Testing to In production.
`gmail.modify` is a restricted scope, so Google shows an unverified-app
warning until the app is verified. The two household accounts can continue
past that warning. Until the screen is in production, each person has to sign
in again every 7 days.

A person who already signed in on the Mac is signed in on the Pi when the
state volume is copied. Anyone who has not signed in yet does it after the
stack is up, from a browser on another machine:

```bash
docker compose --env-file .env -f docker/docker-compose.yml exec -it gateway \
  python -m assistai mail login jacob
```

The command prints a URL. The browser will fail to open `127.0.0.1:8731`.
Paste the full address-bar URL at the `code:` prompt. Repeat for `spouse`.

## Build the image on the Mac

From the repo root, with Docker running:

```bash
make build-pi
docker save assistai/gateway:local -o /tmp/assistai-gateway-arm64.tar
```

The tag `assistai/gateway:local` is replaced by that build. Before loading a
newer image on the Pi, save the one already there so it can be loaded back.

## Copy the tree and the image

On the Pi the compose file expects this layout, with `.env` beside `docker/`:

```text
~/assistai/.env
~/assistai/manifest.toml
~/assistai/config/assistai.toml
~/assistai/docker/docker-compose.yml
~/assistai/docker/searxng/
```

Copy those files from the Mac. `config/assistai.toml` and `.env` are
gitignored. Put the new Fireworks key and a generated SearXNG secret into the
Pi `.env` before the first start:

```bash
openssl rand -hex 32
```

Write the calendar password on the Pi. This is the app-specific password from
appleid.apple.com, the same value the Mac Keychain holds under service
`assistai` and account `caldav`.

```bash
sudo install -d -m 700 /var/lib/assistai
sudo install -m 600 /dev/null /var/lib/assistai/caldav-password
sudoedit /var/lib/assistai/caldav-password
```

Point `.env` at it:

```text
ASSISTAI_CALDAV_PASSWORD_FILE=/var/lib/assistai/caldav-password
```

Compose's `caldav-secret` service copies that file into the volume the
gateway reads and sets mode 600, owner uid 10001.

Load the image:

```bash
docker load -i ~/assistai-gateway-arm64.tar
docker image inspect assistai/gateway:local --format '{{.Os}}/{{.Architecture}} {{.Id}}'
```

The architecture line is `linux/arm64`. The id should match the image just
saved on the Mac.

## Move state, then Signal

Copy the gateway state volume while the Mac stack is still up. It holds
conversation history, list contents, jobs, and Gmail refresh tokens. Leave
`assistai_signal-cli-config` on the Mac. Two running copies of that device
split the Signal session.

On the Mac:

```bash
docker run --rm \
  -v assistai_gateway-state:/from:ro \
  -v /tmp:/to \
  alpine tar czf /to/gateway-state.tar.gz -C /from .
scp /tmp/gateway-state.tar.gz aissist-admin@raspberrypi.local:~/
```

Stop the Mac stack and leave it stopped:

```bash
make down
```

On the Pi, restore the volume, then start. `--no-build` uses the loaded image.
The first start pulls `bbernhard/signal-cli-rest-api:0.100` and
`searxng/searxng:latest`.

```bash
docker volume create assistai_gateway-state
docker run --rm \
  -v assistai_gateway-state:/to \
  -v /home/aissist-admin:/from \
  alpine tar xzf /from/gateway-state.tar.gz -C /to
cd ~/assistai
docker compose --env-file .env -f docker/docker-compose.yml up -d --no-build
```

Link the Pi as a new device. Scan the printed URI from the phone that owns
the bot number: Signal → Settings → Linked devices.

```bash
docker compose --env-file .env -f docker/docker-compose.yml run --rm --no-deps gateway \
  signal link
```

On that phone, remove the Mac from Linked devices. The Mac compose project
stays down. Starting it again would answer the same texts.

## Check, then turn SSH off

### The clock

The gateway logs the household zone, the local time, and the UTC offset as
`clock.zone`. Before it starts Signal it waits up to two minutes for a
believable clock: the kernel reports the clock synchronized, and the clock
is not more than five minutes behind the newest time in the state volume.

```bash
docker compose --env-file .env -f docker/docker-compose.yml logs gateway | grep '"clock\.'
```

`clock.zone` names `America/Chicago` and today's offset: `-06:00` in winter,
`-05:00` in summer. Restart the gateway and run the same command. The new
lines end with `clock.synced`.

```bash
docker compose --env-file .env -f docker/docker-compose.yml restart gateway
```

Check that the zone data in the image knows the daylight saving changes. The
second 1:30 AM on 1 November 2026 is CST; 3:30 AM on 14 March 2027 is CDT.

```bash
docker compose --env-file .env -f docker/docker-compose.yml exec gateway python -c \
  'from datetime import datetime as d; from zoneinfo import ZoneInfo as Z; z = Z("America/Chicago"); print(d(2026, 11, 1, 1, 30, tzinfo=z).strftime("%:z"), d(2026, 11, 1, 1, 30, fold=1, tzinfo=z).strftime("%:z"), d(2027, 3, 14, 3, 30, tzinfo=z).strftime("%:z"))'
```

It prints `-05:00 -06:00 -05:00`.

After every restart, look for these warnings. `clock.unsynced` comes after
two minutes on a clock that was never believed, and its `reason` says which
case it was. If the kernel had not synchronized the clock but it was not
behind the stored history, Signal started anyway. If the kernel did not say
synchronized and the clock was still more than five minutes behind the
stored history, the reason says Signal is not starting: the gateway exits
with status 2 and Docker restarts it into another two-minute wait. A
container that keeps restarting with that line has a Pi clock that NTP is
not setting. Check NTP on the Pi (`timedatectl status`) in either case.
`clock.sync_unknown` means the container could not
ask the kernel, so only the stored history was checked. `clock.store_in_future`
means the kernel said the clock was synchronized but a stored time is more
than a day ahead of it; that row was ignored. If the Pi's clock really is
that far behind, `timedatectl status` will show it. Without a synced kernel
the row is not ignored: Signal waits the two minutes, then refuses to start
until the kernel reports the clock synchronized.

### Jobs

`store jobs` lists each active job with its next runs in household time. It
prints names and times, never a job's prompt.

```bash
docker compose --env-file .env -f docker/docker-compose.yml exec gateway \
  python -m assistai store jobs --next 3
```

Each daily job shows the hour it was asked for. `--from 2027-03-12 --next 4`
starts the list at midnight on that date, which shows a daily job across the
spring daylight saving change on March 14, 2027.

The first start of a schema 15 image fills in a schedule for every job row,
cancelled ones included, and logs one line per row:

```bash
docker compose --env-file .env -f docker/docker-compose.yml logs gateway \
  | grep store.job_schedule_migrated
```

### Texts

From each allowed phone, text the bot. Each assistant should still know the
recent relay and digest that were in the copied state. Calendar and mail
should answer. A stranger gets silence.

Reboot the Pi:

```bash
sudo reboot
```

After it comes back, `docker ps` shows the stack, and both phones still get
an answer. That reboot is the Phase 13 check. A week unattended is the rest
of it.

Then turn SSH off:

```bash
sudo systemctl disable --now ssh
```

## Install a later image

Plug in a keyboard and monitor.

```bash
sudo systemctl enable --now ssh
```

On the Mac, `make build-pi`, save a tar, and copy it. On the Pi, save the
current image first, and keep a tag on it so it can still be run after the
new one is loaded:

```bash
docker save assistai/gateway:local -o ~/assistai-gateway-previous.tar
docker tag assistai/gateway:local assistai/gateway:previous
docker load -i ~/assistai-gateway-arm64.tar
```

If the release notes say the image changes the state database schema,
rehearse on a copy first. Wait until no proposal is waiting for a yes, then
copy the volume while the stack runs:

```bash
docker volume create assistai_state-rehearsal
docker run --rm -v assistai_gateway-state:/from:ro -v assistai_state-rehearsal:/to \
  alpine sh -c 'cp -a /from/. /to/'
docker run --rm -e ASSISTAI_TIMEZONE=America/Chicago -e ASSISTAI_STATE_DIR=/app/state \
  -v assistai_state-rehearsal:/app/state assistai/gateway:local store jobs --next 3
docker run --rm -e ASSISTAI_TIMEZONE=America/Chicago -e ASSISTAI_STATE_DIR=/app/state \
  -v assistai_state-rehearsal:/app/state assistai/gateway:local \
  store jobs --from 2027-03-12 --next 4
```

These need only the zone, so `.env` and its secrets stay out of them. Use
the zone `.env` names.

Every job shows the hour it was asked for, including March 14 and 15, 2027.
Then rehearse the rollback on the same copy, and open it with the previous
image:

```bash
docker run --rm -e ASSISTAI_STATE_DIR=/app/state \
  -v assistai_state-rehearsal:/app/state assistai/gateway:local store downgrade --to 14
docker run --rm -v assistai_state-rehearsal:/app/state --entrypoint python \
  assistai/gateway:previous -c \
  'from pathlib import Path; from assistai.store import Store; s = Store(Path("/app/state/assistai.sqlite")); print([j.name for a in ("jacob", "spouse") for j in s.list_jobs(a)])'
docker volume rm assistai_state-rehearsal
```

The last command prints the job names. Do not install the new image until
it does. Then:

```bash
docker compose --env-file .env -f docker/docker-compose.yml up -d --no-build
```

Run the clock and jobs checks above, then:

```bash
sudo systemctl disable --now ssh
```

`.env`, `config/assistai.toml`, `/var/lib/assistai/caldav-password`, and the
`assistai_gateway-state` volume stay on the Pi. A new image does not replace
them.

## Roll back a schema change

The previous image is the rollback, but an image refuses a state file newer
than its own schema: the gateway exits with "state database is newer than
this build". Step the file back with the newer image first, while it is
still loaded:

```bash
docker compose --env-file .env -f docker/docker-compose.yml stop gateway
docker compose --env-file .env -f docker/docker-compose.yml run --rm --no-deps gateway \
  store downgrade --to 14
docker load -i ~/assistai-gateway-previous.tar
docker compose --env-file .env -f docker/docker-compose.yml up -d --no-build
```

`store downgrade --to 14` changes only the schema version, and only when the
file is exactly what the newer image writes. Schema 15 added nullable
columns the previous image never reads. If the command refuses, the file has
something the previous image may misread: do not load the older image. Start
the newer one again with `up -d --no-build`.

While rolled back, the previous image runs jobs the old way. Installing the
newer image again fills in a schedule for any job created in between, and
rebuilds the schedule of any job the previous image ran or rescheduled
(`store.job_schedule_rebuilt` in the log); check them with `store jobs`.
