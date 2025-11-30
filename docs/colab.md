# Colab GPU Sessions Inside VS Code

This guide explains how we connect VS Code notebooks to Google Colab runtimes so everyone can launch GPU-backed workspaces directly from the repo. Follow the steps below the first time you configure a machine, then reuse the “Daily flow” section whenever you need a new session.

## Required VS Code Setup (one-time per machine)

1. Install VS Code (1.95 or newer) and sign in with your Microsoft/GitHub account if you want settings sync.
2. Install these extensions (they’ll be auto‑recommended by this repo as well):
   - `ms-python.python`
   - `ms-toolsai.jupyter`
   - `google.colab`
3. Command Palette → `Colab: Sign in to Google`, then finish the OAuth device-code flow in your browser. Approve any prompts so VS Code can manage Colab runtimes on your behalf.
4. Optional but recommended: turn on VS Code Settings Sync so the Colab extension is enabled everywhere you log in.

## Daily Flow (per Colab session)

1. Command Palette → `Colab: Connect to Colab`.
2. Pick the Google account, workspace/project, machine type (GPU/TPU/CPU), and region. VS Code lists the live runtime in the Colab sidebar.
3. Open (or create) the notebook you want to run—example: `docs/test.ipynb`.
4. Click `Select Kernel` in the notebook toolbar and choose the kernel that begins with `Colab • Python 3` (the hardware label, e.g., `Tesla T4`, appears next to it).
5. Run cells as usual. They execute in the Colab VM while the notebook stays in your repo workspace. Use a quick sanity check such as:

   ```python
   !nvidia-smi
   ```

   to confirm you’re talking to the GPU.

## Tips for Consistent Environments

- **Bootstrap the runtime** – A Colab VM is clean on every start. Keep a setup cell at the top of important notebooks that clones this repo (or installs it via `pip install -e .`) and installs any extra wheels you need.
- **Shareable dependencies** – If you maintain a Wheel/SDist internally, publish it to your private index so notebooks can install via `pip install sacp-suite --extra-index-url ...`.
- **Persist anything important** – Files saved under `/content` vanish when the runtime shuts down. Push results to git, Drive, or object storage when you’re done.
- **Manage sessions** – Use the Colab sidebar or `Colab: Manage Sessions` command to end idle runtimes quickly. Sessions time out automatically (~90 minutes idle, ~12 hours max).
- **Team-wide defaults** – Because `.vscode/extensions.json` (added in this repo) lists the required extensions, VS Code prompts every contributor to install the Colab toolset, keeping behavior consistent across machines.

Ping the team if Colab adds new hardware types or if quotas change, and update this doc so everyone stays in sync.
