# Using Meta's Muse Code (`muse` CLI) in CERES

Instructions for AI agents and developers working in this repo. CERES is a Meta Global AI Developer Hackathon project and must visibly use Meta's models. Muse Code is Meta's coding agent and runs on Meta's models. Use it for development work in this repo.

## 1. Install (Windows, PowerShell)

```powershell
irm https://dev.meta.ai/install.ps1 | iex
```

What this does (verified 2026-10-26):
- Downloads the launcher from `https://api.meta.ai/muse-launcher.ps1` (HTTPS only, checksum-verified).
- Installs to `%LOCALAPPDATA%\Programs\muse\` (`muse.cmd` and a hidden `.muse-launcher.ps1`).
- Adds that folder to the **user** PATH. Existing terminals won't see it until restarted. To use it in the current session:
  ```powershell
  $env:Path = "$env:LOCALAPPDATA\Programs\muse;$env:Path"
  ```
- The launcher updates itself and the binary automatically (about hourly). Set `MUSE_NO_AUTO_UPDATE=1` to turn this off.

macOS/Linux use the `.sh` installer instead (`installer.ps1` refuses to run on other platforms).

Check the install with `muse --version` (installed version: `Muse Code 1.4.13`).

## 2. Authenticate (needs the human)

Use either method:

- **Meta account login:** run `muse login`. It opens `auth.meta.com` in a browser and shows a code; the **user** must approve it. Credentials are saved to `~\.config\muse\auth.json`.
  - An agent cannot finish this step. Ask the user to run `muse login` in their own terminal.
  - Before `muse` commands, run `muse usage` to check whether you're logged in (`not logged in` means you aren't).
- **API key:** the `META_API_KEY` environment variable (from the Meta Model API dashboard) **always overrides** the account login. Alternatively store it with:
  ```powershell
  Get-Content key.txt | muse auth set --api-key-stdin
  ```
  Never pass keys as command-line arguments, and never commit them. `.env` is gitignored; `.env.example` lists the variable names.

In non-interactive shells (e.g. an agent's tool calls), set `$env:MUSE_LOGIN = '0'` so `muse` fails immediately with an error instead of waiting for a browser login.

## 3. Usage

Interactive (for the human): `muse` or `muse "your prompt"`.

Headless (for agents and scripts): use `muse exec`:

```powershell
muse exec "Explain the structure of this repo"
muse exec --json "..."                     # JSONL events on stdout
muse exec --prompt-file prompt.md          # long prompts
muse exec --image diagram.png "..."        # attach images (repeatable)
muse exec --output-schema schema.json "..."  # structured final answer
muse exec --model <id> --reasoning-effort medium "..."
```

Useful flags:
- `--model <id>`: choose the Meta model (see https://dev.meta.ai/docs/models). `muse model-profile` shows what a model resolves to.
- `--reasoning-effort none|minimal|low|medium|high|xhigh|max|ultra` (default `high`).
- `--approval-mode untrusted|on-request|never` (default `on-request`). Approval prompts and the OS sandbox are **on by default**; keep them on. Don't use `--yolo` / `--disable-sandbox` unless the user asks.
- `--max-model-steps <N>`: cap long runs.
- `-w` / `--worktree create`: run in an isolated git worktree.

Other commands: `muse resume --last`, `muse export` (save a transcript), `muse voice` (audio transcription), `muse usage` (remaining allowance), `muse init` (scaffold agent config).

## 4. Hackathon rules to follow

- **Prove Meta usage:** when Muse Code (or any Meta model) is used to build a feature, record it in the README "Meta Models Used" table and the HACKATHON.md build log. Exported transcripts (`muse export`) are good evidence.
- **New work only:** all code must be written during the hackathon (2026-10-26 to 2026-11-04). Don't paste in pre-existing code; give third-party code a license entry in HACKATHON.md.
- **Repo stays public;** never commit secrets.
- **Meta Model API credits:** $150 per participant. Keep an eye on spend with `muse usage` and the Billing page.

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `muse` not recognized | Restart the terminal, or prepend the PATH line from §1 |
| `not logged in: run muse login first` | Ask the user to run `muse login`, or set `META_API_KEY` |
| Command hangs | Probably waiting for browser login; set `MUSE_LOGIN=0` in automated shells |
| `installed binary is missing` | Rerun the installer from §1 |
