"""Check that PyTorch can use your NVIDIA GPU and that FinBERT runs on it.

   python check_gpu.py
"""
import sys
import time

try:
    import torch
except ImportError:
    sys.exit("FAIL: PyTorch is not installed. See docs/ALPACA_SETUP.md, section 'GPU'.")

print(f"PyTorch {torch.__version__}, built for CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("FAIL: torch.cuda.is_available() is False. Usually this means the CPU-only PyTorch build is installed "
             "(version string ends in '+cpu'), or the NVIDIA driver is too old. See docs/ALPACA_SETUP.md, section 'GPU'.")
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 1e9:.1f} GB memory")

x = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    x @ x
torch.cuda.synchronize()
print(f"Half-precision matrix test OK ({time.time() - t:.2f}s)")

try:
    from strategies.news_sentiment.scorer import finbert_scorer
except ImportError as e:
    sys.exit(f"transformers missing ({e}). Run: pip install -r requirements-news.txt")
print("Loading FinBERT (first run downloads ~440 MB) ...")
score = finbert_scorer(device="cuda", batch_size=128)
texts = ["Company beats earnings expectations and raises full-year guidance",
         "Regulator opens investigation into the company's accounting",
         "Company schedules its annual shareholder meeting for May"]
for tx, s in zip(texts, score(texts)):
    print(f"  {s:+.2f}  {tx}")
big = ["Shares fall after the company cuts its profit forecast"] * 2048
torch.cuda.reset_peak_memory_stats(); t = time.time()
score(big)
dt = time.time() - t
print(f"Speed test: {len(big) / dt:,.0f} headlines/second, peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB of {p.total_memory / 1e9:.1f} GB")
print("GPU is ready. (Sensible output above: good news positive, bad news negative, scheduling about 0.)")
