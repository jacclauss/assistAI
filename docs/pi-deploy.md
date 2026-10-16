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
| Calendar | `.env` username, calendar name, timezone. Password in `/var/lib/assistai/caldav-password` | The Mac reads the Keychain. The Pi reads only that file. |
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
current image first, `docker load` the new tar, then:

```bash
docker compose --env-file .env -f docker/docker-compose.yml up -d --no-build
sudo systemctl disable --now ssh
```

`.env`, `config/assistai.toml`, `/var/lib/assistai/caldav-password`, and the
`assistai_gateway-state` volume stay on the Pi. A new image does not replace
them.
