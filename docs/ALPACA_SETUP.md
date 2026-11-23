# Setting up Alpaca (paper trading) and your GPU for step 10

Nothing here risks real money. A **paper** account is a simulated account with fake dollars. We only read data in
step 10; `check_alpaca.py` never places an order.

## 1. Create the account and the PAPER keys
1. Go to https://alpaca.markets and sign up (free). If you only want paper trading you do not need to fund anything.
2. In the dashboard, switch to **Paper Trading** (top-left account switcher). Check it says PAPER.
3. On the Paper Trading overview page find **API Keys** -> **Generate New Keys**. You get a **key id** (starts with `PK`)
   and a **secret**. The secret is shown ONCE: copy it somewhere safe now.
4. Never paste keys into chat, a script, or a git commit. They live only in environment variables.

## 2. Give the keys to Python (Windows PowerShell)
```powershell
setx APCA_API_KEY_ID "PKxxxxxxxxxxxxxxxx"
setx APCA_API_SECRET_KEY "your-secret-here"
```
`setx` saves them permanently but only for NEW terminals: **close VS Code / PowerShell and reopen**, then reactivate the venv.
(To use them in the current window only: `$env:APCA_API_KEY_ID="PK..."` and `$env:APCA_API_SECRET_KEY="..."`.)

## 3. Test the connection
```powershell
git pull
pip install -r requirements.txt -r requirements-news.txt
python check_alpaca.py
```
You should see: paper account status ACTIVE, equity about $100,000 (Alpaca's default paper balance; our strategies
use their own $100 books and ignore it), and "News API OK: oldest AAPL article since 2015 ...".
Common failures: HTTP 401/403 = wrong keys or live keys used instead of paper keys; "keys not found" = terminal opened before `setx`.

(Later, for paper-trading orders we will add `pip install alpaca-py` and use `TradingClient(key, secret, paper=True)`.
Not needed yet.)

## 4. GPU (NVIDIA, 8 GB is plenty: FinBERT in half precision uses about 1 GB)
1. Run `nvidia-smi`. If it errors, install/update the NVIDIA driver (GeForce Experience or nvidia.com/drivers).
   Note the "CUDA Version" in the top-right corner: that is the highest CUDA your driver supports.
2. Install the CUDA build of PyTorch, NOT the default CPU one. Use the selector at https://pytorch.org/get-started/locally
   (Stable / Windows / Pip / Python / a CUDA version that is <= your driver's). It gives a line like
   `pip install torch --index-url https://download.pytorch.org/whl/cu124`. (The cuXXX part changes over time; as far as I
   know RTX 50-series cards need the newest CUDA builds, cu128 or later.) If you already have the CPU build, add `--force-reinstall`.
3. Check:
```powershell
python check_gpu.py
```
It must print your GPU name, a sensible score for each of the three sample headlines (good news positive, bad news
negative, scheduling about zero) and a speed in headlines/second. If it says `torch.cuda.is_available() is False`, the
version printed in the first line probably ends in `+cpu`: reinstall step 2.
4. `python step10_news_sentiment.py score` then uses the GPU automatically and refuses to fall back to the CPU
   unless you pass `--cpu`. If you run out of GPU memory (unlikely), use `--batch 64`.
