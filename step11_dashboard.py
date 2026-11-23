"""STEP 11 - Dashboard: all strategies side by side, each starting from $100, TRAIN PERIOD ONLY.

   python step11_dashboard.py          # writes output/dashboard.html - open it in your browser

Nothing after 2025-12-31 is read for returns, so no test look is used. Pairs is not included (no edge, and its yearly
cointegration search is slow); sector rotation is shown greyed out as rejected and left out of the combined book.
"""
import base64
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from engine import metrics
from lab import registry
from lab.portfolio import (combine_equal, combine_inverse_vol, common_window, lookthrough, risk_contributions,
                           rolling_avg_correlation)
from lab.stats import alpha_beta

OUT = Path("output") / "dashboard.html"
# one fixed colour per entity (validated categorical palette, never re-assigned by rank)
COLORS = {"momentum": "#2a78d6", "trend": "#eb6834", "sma": "#1baf7a", "fx_carry": "#eda100",
          "sector": "#8a8984", "spy": "#b5b4ae", "equal": "#0b0b0b", "invvol": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def shade_crises(ax):
    for name, (a, b) in config.CRISIS_WINDOWS.items():
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#000000", alpha=0.05, linewidth=0)


def label_ends(ax, series, colors):
    """Direct labels at the right edge, nudged apart so they do not overlap."""
    items = sorted(((s.dropna().iloc[-1], n) for n, s in series.items()))
    lo, hi = ax.get_ylim()
    gap = (np.log(hi) - np.log(lo)) * 0.045 if ax.get_yscale() == "log" else (hi - lo) * 0.045
    ys, prev = [], None
    for v, n in items:
        y = np.log(v) if ax.get_yscale() == "log" else v
        y = y if prev is None or y - prev >= gap else prev + gap
        ys.append((n, y)); prev = y
    for n, y in ys:
        yy = np.exp(y) if ax.get_yscale() == "log" else y
        ax.text(1.01, yy, n, transform=ax.get_yaxis_transform(), color=colors[n], fontsize=9, va="center", fontweight="bold")


def to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pct(x, d=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}%}"


def num(x, d=2):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{d}f}"


def main():
    books, warnings = registry.build_books()
    if len(books) < 2:
        raise SystemExit("Fewer than two strategies could be built:\n" + "\n".join(warnings))
    spy = registry.spy_returns()
    df = common_window({k: b["ret"] for k, b in books.items()})
    df_spy = df.join(spy.rename("spy"), how="inner").dropna()
    df = df_spy.drop(columns="spy")
    active = [k for k in df.columns if books[k]["active"]]
    eq = combine_equal(df[active])
    iv = combine_inverse_vol(df[active])
    names = {k: books[k]["label"] for k in books}
    colors_by_label = {names[k]: COLORS[k] for k in books}
    colors_by_label.update({"SPY": COLORS["spy"], "Combined (equal $)": COLORS["equal"]})

    # ---------- scoreboard ----------
    rows = []
    for k, b in books.items():
        r, own = df[k], metrics(b["ret"])
        m = metrics(r)
        fair = alpha_beta(r, b["bench"].reindex(r.index)) if b["bench"] is not None else {}
        vs_spy = alpha_beta(r, df_spy["spy"])
        rows.append((k, b, m, own, fair, vs_spy))
    comb_rows = [("Combined (equal $, active books)", metrics(eq), None), ("Combined (inverse-vol, shorter window)", metrics(iv), iv)]

    # ---------- charts ----------
    figs = {}
    eqty = (1 + df_spy).cumprod() * 100
    eqty["comb"] = (1 + eq.reindex(df_spy.index)).cumprod() * 100
    fig, ax = plt.subplots(figsize=(9.5, 4.8)); style(ax); ax.set_yscale("log"); shade_crises(ax)
    series = {}
    for k in books:
        ax.plot(eqty.index, eqty[k], color=COLORS[k], lw=1.8 if books[k]["active"] else 1.2, alpha=1.0 if books[k]["active"] else 0.85)
        series[names[k]] = eqty[k]
    ax.plot(eqty.index, eqty["spy"], color=COLORS["spy"], lw=1.4); series["SPY"] = eqty["spy"]
    ax.plot(eqty.index, eqty["comb"], color=COLORS["equal"], lw=2.4); series["Combined (equal $)"] = eqty["comb"]
    ax.set_title(f"Growth of $100, {eqty.index[0].date()} to {eqty.index[-1].date()} (log scale; grey bands = crisis windows)", fontsize=10, color=INK, loc="left")
    fig.canvas.draw(); label_ends(ax, series, colors_by_label)
    figs["equity"] = to_img(fig)

    dd = lambda s: (1 + s).cumprod() / (1 + s).cumprod().cummax() - 1
    fig, ax = plt.subplots(figsize=(9.5, 3.6)); style(ax); shade_crises(ax)
    for k in active:
        ax.plot(df.index, dd(df[k]), color=COLORS[k], lw=1.4)
    ax.plot(eq.index, dd(eq), color=COLORS["equal"], lw=2.2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Drawdown: how far below its previous peak each active book is", fontsize=10, color=INK, loc="left")
    ax.legend([names[k] for k in active] + ["Combined (equal $)"], fontsize=8, frameon=False, ncol=3, loc="lower left")
    figs["drawdown"] = to_img(fig)

    cm = df_spy.rename(columns={**names, "spy": "SPY"}).corr()
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(cm.values, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cm))); ax.set_xticklabels(cm.columns, rotation=35, ha="right", fontsize=9, color=MUTED)
    ax.set_yticks(range(len(cm))); ax.set_yticklabels(cm.index, fontsize=9, color=MUTED)
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=9, color=INK if abs(cm.values[i, j]) < 0.6 else "white")
    ax.set_title("Correlation of daily returns (1 = move together, 0 = unrelated)", fontsize=10, color=INK, loc="left")
    figs["corr"] = to_img(fig)

    rc = rolling_avg_correlation(df[active], 252)
    fig, ax = plt.subplots(figsize=(9.5, 3.0)); style(ax); shade_crises(ax)
    ax.plot(rc.index, rc.values, color=COLORS["equal"], lw=1.8)
    ax.axhline(0, color=GRID)
    ax.set_title("Average correlation between the active books, trailing 1 year", fontsize=10, color=INK, loc="left")
    figs["rolling"] = to_img(fig)

    contrib = risk_contributions(df[active])
    fig, ax = plt.subplots(figsize=(6.2, 2.6)); style(ax)
    ax.barh([names[k] for k in active], [contrib[k] for k in active], color=[COLORS[k] for k in active], height=0.55)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    for i, k in enumerate(active):
        ax.text(contrib[k] + 0.01, i, f"{contrib[k]:.0%}", va="center", fontsize=9, color=INK)
    ax.set_title("Share of the combined book's risk (each book gets 25% of the dollars)" if len(active) == 4 else "Share of combined risk (equal dollars)", fontsize=10, color=INK, loc="left")
    figs["risk"] = to_img(fig)

    # ---------- look-through at the train end ----------
    holdings = {names[k]: books[k]["weights"] for k in books}
    groups = {}
    for k in books:
        groups.update(books[k]["groups"])
    lt = lookthrough(holdings, groups)
    tech = {names[k]: sum(v for g, v in lt[names[k]].items() if g in registry.TECH_GROUPS or g == "technology") for k in books}

    # ---------- HTML ----------
    def table(header, body):
        return "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in header) + "</tr></thead><tbody>" + "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in body) + "</tbody></table>"

    sc = []
    for k, b, m, own, fair, vs_spy in rows:
        fair_txt = "vs cash" if b["bench"] is None else (f"{fair['alpha']:+.1%} (t {fair['t_alpha']:.1f})" if fair else "n/a")
        sc.append([f'<span class="dot" style="background:{COLORS[k]}"></span>{html.escape(b["label"])}', html.escape(b["status"]),
                   f"{b['ret'].index[0].date()}", num(m["sharpe"]), num(own["sharpe"]), pct(m["cagr"]), pct(m["volatility"]), pct(m["max_drawdown"]),
                   fair_txt + (f"<br><small>vs {html.escape(b['bench_name'])}</small>" if b["bench"] is not None else ""),
                   num(vs_spy.get("beta")) if vs_spy else "n/a"])
    for label, m, series in comb_rows:
        sc.append([f"<b>{html.escape(label)}</b>", "", "", num(m["sharpe"]), "", pct(m["cagr"]), pct(m["volatility"]), pct(m["max_drawdown"]), "", ""])
    score_html = table(["Strategy", "Verdict", "Starts", "Sharpe (common window)", "Sharpe (own window)", "CAGR", "Volatility", "Worst drawdown",
                        "Alpha vs fair benchmark", "Beta to SPY"], sc)

    lt_show = lt[(lt.abs() > 0.005).any(axis=1)].copy()
    order = [g for g in lt_show.index if g != "cash"] + (["cash"] if "cash" in lt_show.index else [])
    lt_html = table(["Group"] + list(lt_show.columns), [[html.escape(g)] + [pct(lt_show.loc[g, c], 0) if abs(lt_show.loc[g, c]) > 0.005 else "" for c in lt_show.columns] for g in order]
                    + [["<b>Technology-like total</b>"] + [f"<b>{tech[c]:.0%}</b>" for c in lt_show.columns]])
    avg_corr = (df[active].corr().values.sum() - len(active)) / (len(active) * (len(active) - 1)) if len(active) > 1 else float("nan")
    vols = df[active].std() * np.sqrt(252)
    div_ratio = float((vols.mean()) / (eq.std() * np.sqrt(252)))
    warn_html = "".join(f"<li>{html.escape(w)}</li>" for w in warnings) or "<li>none</li>"

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Quant dashboard (train period)</title>
<style>
body{{font:15px/1.5 system-ui,Segoe UI,Arial,sans-serif;color:{INK};background:#fcfcfb;max-width:1050px;margin:24px auto;padding:0 16px}}
h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:34px 0 6px}} .sub{{color:{MUTED}}}
.banner{{background:#fff4d6;border:1px solid #eda100;border-radius:8px;padding:10px 14px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid {GRID};padding:6px 8px;text-align:left;vertical-align:top}}
th{{color:{MUTED};font-weight:600}} .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}}
img{{max-width:100%;height:auto}} .note{{color:{MUTED};font-size:13px;margin:4px 0 10px}} small{{color:{MUTED}}}
.two{{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}}
</style></head><body>
<h1>Quant research dashboard</h1>
<div class="sub">Each strategy starts from $100. Train period only: {eqty.index[0].date()} to {eqty.index[-1].date()} on the common window.</div>
<div class="banner"><b>Test period locked.</b> Nothing after {config.TRAIN_END.date()} is used for any number or chart here, and no test look is consumed.
Research and paper-trading methodology only: not investment advice, and the dollar figures are simulated.</div>

<h2>1. Scoreboard</h2>
<p class="note">Sharpe = return per unit of risk, measured above what T-bills paid. "Common window" puts every strategy on the same dates so they are comparable;
"own window" is each strategy's full history (matches the earlier steps). Alpha vs fair benchmark: the part of the return NOT explained by simply owning the
strategy's own yardstick; |t| of about 2 or more is needed before it looks like more than luck. Beta to SPY: 1.0 = moves one-for-one with the US stock market.</p>
{score_html}
<p class="note">Rejected books (thin grey line) are shown for reference and excluded from the combined book. Pairs is not shown (no edge found).</p>

<h2>2. Growth of $100</h2>
<p class="note">Log scale: equal vertical distances are equal percentage moves. Look at how the books behave in the grey crisis bands, not just where they end.</p>
<img src="{figs['equity']}" alt="Growth of $100 for each strategy, SPY and the combined book">

<h2>3. Drawdowns</h2>
<p class="note">The pain you would have lived through: distance below the previous peak. A diversifier earns its place here, not in the average return.</p>
<img src="{figs['drawdown']}" alt="Drawdowns of the active books and the combined book">

<h2>4. Are the strategies secretly the same bet?</h2>
<div class="two"><div><img src="{figs['corr']}" alt="Correlation matrix"></div>
<div style="flex:1;min-width:300px"><p class="note">Average correlation between the active books: <b>{avg_corr:.2f}</b>. The combined book's risk is
<b>{1 / div_ratio:.0%}</b> of the average single-book risk (lower = more diversification).
Momentum and SMA both trade the same stock list, so a high correlation between them means two names for one bet, not two bets. Low or negative numbers
(trend, FX carry) are where the diversification comes from.</p></div></div>
<img src="{figs['rolling']}" alt="Rolling average correlation">
<p class="note">If this line climbs toward 1 inside the grey crisis bands, the diversification disappears exactly when it is needed.</p>

<h2>5. Equal dollars are not equal risk</h2>
<img src="{figs['risk']}" alt="Risk contribution of each active book">
<p class="note">A book with higher volatility drives the combined book's ups and downs even with the same dollars. Weights are NOT optimised on purpose:
picking the weights that look best on the training period would be another round of fitting.</p>

<h2>6. What the books hold at the end of the train period ({config.TRAIN_END.date()})</h2>
<p class="note">Look-through by group, to answer "is everything quietly long technology?". Groups: semiconductors, software/internet and hardware/networking
stocks plus the XLK ETF count as technology-like. FX carry's table column is currency exposure; trend's is by asset class.</p>
{lt_html}

<h2>7. Read this before drawing conclusions</h2>
<ul class="note">
<li>Survivorship bias: the stock lists are today's biggest companies, so the stock strategies (momentum, SMA) and their benchmarks are flattered.</li>
<li>Correlations and Sharpe ratios come from one history. A 1-year Sharpe has a standard error of about 1, and even 20 years leaves about 0.2.</li>
<li>The combined book is an illustration at equal dollars, rebalanced daily (real rebalancing would cost something). It is not a recommendation.</li>
<li>Settings are the fixed v1 choices from the individual steps; each was chosen on this same train period, so in-sample results are optimistic.</li>
<li>Data warnings for this run:</li>{warn_html}
</ul></body></html>"""
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(page, encoding="utf-8")
    print(f"Wrote {OUT.resolve()}  ({len(books)} books, {len(df)} common days, active: {', '.join(names[k] for k in active)})")
    for w in warnings:
        print("WARNING:", w)
    print(f"Average correlation between active books {avg_corr:.2f}; combined risk is {1 / div_ratio:.0%} of the average single book.")
    print("Technology-like share of holdings at train end:", {k: f"{v:.0%}" for k, v in tech.items()})


if __name__ == "__main__":
    main()
