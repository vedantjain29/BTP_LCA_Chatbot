"""
fetch_fulltext.py - download legal open-access PDFs for the papers in the dataset.

No API key needed. Uses:
  * arXiv direct PDF links
  * direct .pdf links already present in the dataset URL
  * Crossref (title -> DOI)   [free, just your email]
  * Unpaywall (DOI -> open-access PDF location)   [free, just your email]

Run from the same folder as your other scripts:

    python fetch_fulltext.py --email you@example.com --dry-run     # no network, shows the plan
    python fetch_fulltext.py --email you@example.com               # core scope (default)
    python fetch_fulltext.py --email you@example.com --scope all

Outputs (in ../data/):
    pdfs/<ID>.pdf                 downloaded PDFs, named by the dataset ID
    fulltext_report.csv           one row per in-scope paper with its status
    manual_download_needed.csv    papers you must download yourself (save as pdfs/<ID>.pdf)

Re-running is safe: any pdfs/<ID>.pdf that already exists (including ones you add by
hand) is kept and reported as "already_have".
"""
import argparse
import csv
import difflib
import re
import time
from pathlib import Path
from urllib.parse import unquote

import requests

# --- scope ---------------------------------------------------------------------------
CORE = (
    "Renewable Electricity",      # matches every "Renewable Electricity - ..." subdomain
    "LCA Methodology",
    "LCI Databases",
    "Uncertainty/Data Quality",
    "ML for LCI",
)
EXTENDED = CORE + (
    "Green Hydrogen",
    "Batteries/EV",
    "Water/Biodiversity Footprint",
    "Social LCA/LCSA",
)


def in_scope(subdomain: str, scope: str) -> bool:
    if scope == "all":
        return True
    prefixes = CORE if scope == "core" else EXTENDED
    return any(subdomain.startswith(p) for p in prefixes)


# --- URL helpers (pure functions, no network) ------------------------------------------
DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s?#\"<>]+)", re.I)
ARXIV_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf|html)/((?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)",
    re.I,
)
NATURE_RE = re.compile(r"nature\.com/articles/(s\d{5}-\d{3}-\d{5}-\w)", re.I)
TRAILING_PAGE = re.compile(r"/(full|pdf|epdf|abstract|fulltext|html|read)(/.*)?$", re.I)


def arxiv_pdf_url(url: str):
    m = ARXIV_RE.search(url)
    if not m:
        return None
    arxiv_id = re.sub(r"\.pdf$", "", m.group(1), flags=re.I)
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def doi_from_url(url: str):
    url = unquote(url)
    m = NATURE_RE.search(url)
    if m:                                    # Nature/Scientific Reports: DOI = 10.1038/<article id>
        return f"10.1038/{m.group(1)}"
    m = DOI_RE.search(url)
    if not m:
        return None
    doi = TRAILING_PAGE.sub("", m.group(1)).rstrip(".,;)")
    return doi


def is_direct_pdf(url: str) -> bool:
    return bool(re.search(r"\.pdf($|\?)", url, re.I))


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def plan_route(url: str):
    """Which route will we try first? Used by --dry-run and by the real run."""
    if arxiv_pdf_url(url):
        return "arxiv_direct"
    if is_direct_pdf(url):
        return "direct_pdf"
    if doi_from_url(url):
        return "doi_in_url"
    return "needs_title_lookup"


# --- network helpers ----------------------------------------------------------------------
class Fetcher:
    def __init__(self, email: str):
        self.email = email
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": f"BTP-LCA-Chatbot/0.1 (mailto:{email}) python-requests"}
        )

    def crossref_doi(self, title: str):
        try:
            r = self.session.get(
                "https://api.crossref.org/works",
                params={"query.bibliographic": title, "rows": 3,
                        "select": "DOI,title", "mailto": self.email},
                timeout=30,
            )
            if r.status_code != 200:
                return None
            best, best_ratio = None, 0.0
            for item in r.json().get("message", {}).get("items", []):
                for cand in item.get("title", []) or []:
                    ratio = difflib.SequenceMatcher(None, norm_title(title), norm_title(cand)).ratio()
                    if ratio > best_ratio:
                        best, best_ratio = item.get("DOI"), ratio
            return best if best_ratio >= 0.90 else None
        except requests.RequestException:
            return None

    def unpaywall_pdf_urls(self, doi: str):
        try:
            r = self.session.get(
                f"https://api.unpaywall.org/v2/{doi}",
                params={"email": self.email}, timeout=30,
            )
            if r.status_code != 200:
                return []
            data = r.json()
            locs = []
            if data.get("best_oa_location"):
                locs.append(data["best_oa_location"])
            locs += data.get("oa_locations") or []
            urls = []
            for loc in locs:
                u = loc.get("url_for_pdf")
                if u and u not in urls:
                    urls.append(u)
            return urls
        except (requests.RequestException, ValueError):
            return []

    def download_pdf(self, url: str, dest: Path) -> bool:
        """Download and verify it really is a PDF. Returns True on success."""
        tmp = dest.with_suffix(".part")
        try:
            with self.session.get(url, timeout=60, stream=True, allow_redirects=True) as r:
                if r.status_code != 200:
                    return False
                size = 0
                first = b""
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if not first:
                            first = chunk[:5]
                            if first != b"%PDF-":
                                break                 # HTML landing page / paywall, not a PDF
                        size += len(chunk)
                        if size > 80 * 1024 * 1024:   # sanity cap: 80 MB
                            break
                        f.write(chunk)
            ok = first == b"%PDF-" and 30_000 < size <= 80 * 1024 * 1024
            if ok:
                tmp.replace(dest)
            else:
                tmp.unlink(missing_ok=True)
            return ok
        except requests.RequestException:
            tmp.unlink(missing_ok=True)
            return False


# --- main -------------------------------------------------------------------------------
def run(input_csv, out_dir, email, scope, dry_run, limit):
    out_dir = Path(out_dir)
    pdf_dir = out_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    with open(input_csv, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if in_scope(r["Subdomain"].strip(), scope)]
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} papers in scope '{scope}'\n")

    if dry_run:
        counts = {}
        for r in rows:
            route = plan_route(r["URL"])
            counts[route] = counts.get(route, 0) + 1
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {k:20s} {v}")
        print("\nDry run only - nothing downloaded.")
        return

    fx = Fetcher(email)
    report, manual = [], []

    for i, r in enumerate(rows, 1):
        pid, title, url = r["ID"].strip(), r["Title"].strip(), r["URL"].strip()
        dest = pdf_dir / f"{pid}.pdf"
        rec = {"ID": pid, "Title": title, "Subdomain": r["Subdomain"], "URL": url,
               "status": "", "source": "", "doi": "", "pdf_url": "", "file": ""}
        print(f"[{i}/{len(rows)}] {pid} {title[:70]}")

        if dest.exists() and dest.stat().st_size > 30_000:
            rec.update(status="already_have", file=str(dest))
            report.append(rec)
            continue

        candidates = []                       # (source, pdf_url)
        a = arxiv_pdf_url(url)
        if a:
            candidates.append(("arxiv", a))
        elif is_direct_pdf(url):
            candidates.append(("direct_link", url))

        doi = doi_from_url(url)
        if not doi and not a:
            doi = fx.crossref_doi(title)
            time.sleep(0.5)
        rec["doi"] = doi or ""
        if doi and not a:
            for u in fx.unpaywall_pdf_urls(doi):
                candidates.append(("unpaywall", u))
            time.sleep(0.5)

        done = False
        for src, u in candidates:
            if fx.download_pdf(u, dest):
                rec.update(status="downloaded", source=src, pdf_url=u, file=str(dest))
                done = True
                break
            time.sleep(1.0)
        if not done:
            if candidates:
                rec.update(status="oa_found_download_failed", pdf_url=candidates[0][1])
            elif doi:
                rec.update(status="no_open_access_found")
            else:
                rec.update(status="doi_not_found")
            landing = f"https://doi.org/{doi}" if doi else url
            manual.append({"ID": pid, "Title": title, "open_this": landing,
                           "save_as": f"{pid}.pdf", "status": rec["status"]})
        report.append(rec)
        time.sleep(3.0 if a else 1.0)         # be polite (arXiv asks for ~3s)

    with open(out_dir / "fulltext_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)
    with open(out_dir / "manual_download_needed.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ID", "Title", "open_this", "save_as", "status"])
        w.writeheader()
        w.writerows(manual)

    have = sum(1 for r in report if r["status"] in ("downloaded", "already_have"))
    print(f"\nFull text for {have} / {len(report)} papers")
    for s in sorted({r["status"] for r in report}):
        print(f"  {s:28s} {sum(1 for r in report if r['status'] == s)}")
    print(f"\nPDFs:    {pdf_dir}")
    print(f"Report:  {out_dir / 'fulltext_report.csv'}")
    print(f"Manual:  {out_dir / 'manual_download_needed.csv'}  ({len(manual)} papers)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", required=True, help="your email (sent to Crossref/Unpaywall, no key needed)")
    ap.add_argument("--input", default="../data/LCA_papers_dataset.csv")
    ap.add_argument("--out", default="../data")
    ap.add_argument("--scope", choices=["core", "extended", "all"], default="core")
    ap.add_argument("--dry-run", action="store_true", help="show the plan without any network calls")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N papers (for testing)")
    args = ap.parse_args()
    run(args.input, args.out, args.email, args.scope, args.dry_run, args.limit)