"""A one-page HTML report of what the edit did, for review before publishing."""
import html


def _t(s):
    return f"{int(s // 60)}:{s % 60:05.2f}"


def write_report(plan, cfg, results, out_path):
    c = cfg["brand"]["colors"]
    s = plan["stats"]
    words = plan["words"]
    rows = "".join(
        f"<tr><td>{_t(cu['start'])}</td><td>{cu['dur']:.2f}s</td><td>{', '.join(cu['reasons'])}</td>"
        f"<td>{html.escape(cu['text'])}</td></tr>" for cu in plan["cuts"])
    zooms = "".join(
        f"<tr><td>{_t(z['start'])}</td><td>{z['type']}</td><td>{z['level']}</td>"
        f"<td>{html.escape(z.get('word', ''))}</td><td>{z.get('fx')}, {z.get('fy')}</td></tr>" for z in plan["zooms"])
    broll = "".join(
        f"<tr><td>{_t(b['start'])}</td><td>{b['id']}</td><td>{html.escape(b['word'])}</td></tr>" for b in plan["broll"])
    caps = "".join(
        f"<tr><td>{_t(g['start'])}</td><td>{html.escape(' '.join(words[i]['text'] for i in g['words']))}</td></tr>"
        for g in plan["captions"])
    files = "".join(f"<li>{html.escape(str(p))}</li>" for ps in results.values() for p in ps)
    removed = ", ".join(f"{k}: {v}" for k, v in s["removed_words"].items()) or "none"
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Edit report</title>
<style>
body{{background:{c['bg']};color:{c['fg']};font:15px/1.5 "Atkinson Hyperlegible","Segoe UI",sans-serif;margin:0;padding:32px}}
h1{{font:800 34px "Bricolage Grotesque",sans-serif;margin:0 0 4px}} h2{{color:{c['accent']};font:700 13px "JetBrains Mono",monospace;letter-spacing:.12em;text-transform:uppercase;margin:32px 0 8px}}
.k{{display:inline-block;background:{c['paper']};border:1px solid {c['rule']};border-radius:12px;padding:10px 16px;margin:6px 8px 0 0}}
.k b{{display:block;font:800 22px "Bricolage Grotesque",sans-serif}} .k span{{color:{c['muted']};font-size:12px}}
table{{border-collapse:collapse;width:100%;background:{c['paper']};border:1px solid {c['rule']}}}
td{{border-bottom:1px solid {c['rule']};padding:5px 10px;vertical-align:top}} td:first-child{{font-family:"JetBrains Mono",monospace;color:{c['muted']};white-space:nowrap}}
</style></head><body>
<h1>{html.escape(plan.get('title') or '')}</h1><div style="color:{c['muted']}">{html.escape(plan.get('source', ''))} · language {plan.get('language')}</div>
<div><div class="k"><b>{_t(s['source_duration'])}</b><span>original</span></div>
<div class="k"><b>{_t(s['edited_duration'])}</b><span>edited</span></div>
<div class="k"><b>{s['removed_pct']}%</b><span>removed</span></div>
<div class="k"><b>{s['cuts']}</b><span>cuts</span></div>
<div class="k"><b>{s['zooms']}</b><span>zooms</span></div>
<div class="k"><b>{s['broll']}</b><span>B-roll cards</span></div></div>
<p>Removed words: {html.escape(removed)}. Untranscribed speech kept: {s['kept_untranscribed']}.</p>
<h2>Files</h2><ul>{files}</ul>
<h2>Cuts</h2><table>{rows}</table>
<h2>Zooms</h2><table>{zooms}</table>
<h2>B-roll</h2><table>{broll}</table>
<h2>Captions</h2><table>{caps}</table>
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")
    return out_path
