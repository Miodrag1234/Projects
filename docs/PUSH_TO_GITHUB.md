# Push this project to GitHub

## Option A — One script (recommended)

In PowerShell, from the project folder:

```powershell
cd c:\udis--d-kod\UnsupervisedDeepImageStitching-main
powershell -ExecutionPolicy Bypass -File scripts\push-to-github.ps1
```

The script will:

1. `git init` (if needed)
2. `git add` (uses `.gitignore` — skips `.venv`, checkpoints, full datasets)
3. Create a commit
4. Ask for your **GitHub repo URL** (after you create an empty repo on github.com/new)
5. `git push -u origin main`

---

## Option B — GitHub CLI (fully automatic create + push)

Install once:

```powershell
winget install GitHub.cli
gh auth login
```

Then from the project folder:

```powershell
cd c:\udis--d-kod\UnsupervisedDeepImageStitching-main
git init -b main
git add -A
git commit -m "Initial portfolio commit"
gh repo create deep-image-stitching-portfolio --public --source=. --remote=origin --push
```

Change `deep-image-stitching-portfolio` and `--public` to `--private` if you prefer.

---

## Option C — Manual (GitHub website)

1. https://github.com/new → create **empty** repo (no README).
2. In PowerShell:

```powershell
cd c:\udis--d-kod\UnsupervisedDeepImageStitching-main
git init -b main
git add -A
git status
git commit -m "Initial portfolio commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

## After the first push (updates)

Whenever you change files:

```powershell
cd c:\udis--d-kod\UnsupervisedDeepImageStitching-main
git add -A
git commit -m "Update results / examples"
git push
```

Or run `scripts\push-to-github.ps1` again.

---

## What will NOT upload (by design)

See `.gitignore`: `.venv/`, `*.ckpt*`, full `training/` / `testing/` / `testing2/` images, large `results/` folders.

What **will** upload: code, English README, `docs/`, `examples/stitched/*.jpg`, `datasets/*/README.md`.

---

## Sync from the other PC

On the **other computer**, clone once:

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

Then `git pull` / `git push` to keep both machines in sync.
