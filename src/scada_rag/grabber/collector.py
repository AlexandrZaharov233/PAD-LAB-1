from __future__ import annotations

import io
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from bs4 import BeautifulSoup
from scada_rag.common import DATA, ROOT, digest, now, read_json, write_json
from scada_rag.preprocessing.text import clean

USER_AGENT = "SCADA-RAG-University-Lab/1.0 (low-rate educational document fetcher)"


def fetch(url, etag=None, modified=None, attempts=3):
    url = urllib.parse.quote(url, safe=":/?=&%+#@")
    headers = {"User-Agent": USER_AGENT}
    if etag:
        headers["If-None-Match"] = etag
    if modified:
        headers["If-Modified-Since"] = modified
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                payload = response.read(40 * 1024 * 1024 + 1)
                if len(payload) > 40 * 1024 * 1024:
                    raise ValueError("Документ превышает 40 МБ")
                return payload, dict(response.headers), response.geturl()
        except urllib.error.HTTPError as exc:
            if exc.code == 304:
                return None, dict(exc.headers), url
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
        except (OSError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(1 + attempt)


def resolve_anre(source):
    raw, _, _ = fetch(source["registry"])
    soup = BeautifulSoup(raw, "html.parser")
    matches = []
    for block in soup.select(".block-activity-report"):
        title = clean(block.select_one(".title").get_text(" ", strip=True))
        if source["match"].casefold() in title.casefold():
            matches += [urllib.parse.urljoin(source["registry"], a["href"])
                        for a in block.select("a.download-link[href]") if ".pdf" in a["href"].lower()]
    if len(set(matches)) != 1:
        raise ValueError("ANRE: не найден единственный PDF №586 от 16.10.2024. Проверьте configs/sources.json")
    return matches[0]


def parse_html(raw):
    soup = BeautifulSoup(raw, "html.parser")
    body = soup.select_one("article") or soup.select_one("main")
    if body is None:
        raise ValueError("Не найдено основное содержимое HTML; меню нельзя индексировать вместо документа")
    for node in body.select("script,style,nav,aside,footer,.breadcrumb,.toc"):
        node.decompose()
    title_node = body.find("h1")
    title = clean(title_node.get_text(" ", strip=True) if title_node else soup.title.get_text())
    section, blocks = title, []
    for node in body.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr", "pre"]):
        # Do not duplicate nested list items, paragraphs inside tables, etc.
        if node.find_parent(["li", "tr", "pre"]):
            continue
        text = clean(node.get_text(" ", strip=True))
        if node.name.startswith("h"):
            section = text
        elif len(text) >= 15:
            blocks.append({"text": text, "section": section, "page": None})
    if len(" ".join(b["text"] for b in blocks)) < 150:
        raise ValueError("Слишком мало текста: возможна страница ошибки или изменение верстки")
    return title, blocks


def tesseract_path():
    candidates = [shutil.which("tesseract"),
                  str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tesseract-OCR/tesseract.exe"),
                  str(Path(os.environ.get("LOCALAPPDATA", "C:/")) / "Programs/Tesseract-OCR/tesseract.exe")]
    return next((x for x in candidates if x and Path(x).exists()), None)


def ocr_setup():
    command = tesseract_path()
    if not command:
        raise RuntimeError("Для скана ANRE нужен Tesseract OCR. Запустите setup.bat или установите Tesseract-OCR")
    folder = ROOT / "models/ocr"
    folder.mkdir(parents=True, exist_ok=True)
    trained = folder / "ron.traineddata"
    if not trained.exists():
        data, _, _ = fetch("https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/ron.traineddata")
        trained.write_bytes(data)
    return command, folder


def parse_pdf(raw, source, log=print):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(raw))
    start, end = source.get("pages", [1, len(reader.pages)])
    if end > len(reader.pages):
        raise ValueError("ANRE: изменился состав PDF; проверьте диапазон страниц")
    result, rendered = [], None
    for n in range(start, end + 1):
        text = clean(reader.pages[n - 1].extract_text() or "")
        method = "pdf_text"
        if len(text) < 80:
            import pypdfium2 as pdfium
            import pytesseract
            command, tessdata = ocr_setup()
            pytesseract.pytesseract.tesseract_cmd = command
            cache = DATA / "ocr_cache" / (digest(raw) + f"-{n}-ron-fast-180dpi.json")
            cached = read_json(cache)
            if cached:
                text = cached["text"]
            else:
                if rendered is None:
                    rendered = pdfium.PdfDocument(raw)
                log(f"OCR ANRE: страница {n}/{end}")
                bitmap = rendered[n - 1].render(scale=2.5)
                picture = bitmap.to_pil()
                text = clean(pytesseract.image_to_string(picture, lang="ron", config=f'--tessdata-dir "{tessdata}" --psm 3', timeout=120))
                picture.close()
                bitmap.close()
                write_json(cache, {"text": text})
            method = "ocr"
        if len(re.findall(r"[A-Za-zĂÂÎȘȚăâîșț]", text)) < 30:
            raise ValueError(f"Страница {n} не распознана; проверьте OCR")
        result.append({"text": text, "section": f"ANRE №586, страница PDF {n}", "page": n, "extraction": method})
    if rendered is not None:
        rendered.close()
    return source["title"], result


def collect(log=print):
    sources = read_json(ROOT / "configs/sources.json")["sources"]
    state = read_json(DATA / "state.json", {})
    results = []
    (DATA / "raw").mkdir(parents=True, exist_ok=True)
    (DATA / "processed").mkdir(parents=True, exist_ok=True)
    for source in sources:
        sid = source["id"]
        old = state.get(sid, {})
        output = DATA / "processed" / f"{sid}.json"
        try:
            url = resolve_anre(source) if source["kind"] == "anre_pdf" else source["url"]
            same_url = old.get("url") == url
            raw, headers, final_url = fetch(url, old.get("etag") if same_url and output.exists() else None,
                                           old.get("last_modified") if same_url and output.exists() else None)
            raw_path = DATA / "raw" / (sid + (".pdf" if source["kind"] == "anre_pdf" else ".html"))
            if raw is None:
                if not raw_path.exists() or not output.exists():
                    raw, headers, final_url = fetch(url)
            if raw is None and old.get("parser_version") != 6:
                raw = raw_path.read_bytes()
            if raw is None or (output.exists() and digest(raw) == old.get("sha256") and old.get("parser_version") == 6):
                status = "unchanged"
            else:
                snapshot = read_json(DATA / "snapshots" / f"{sid}.json")
                if snapshot and snapshot.get("sha256") == digest(raw) and snapshot.get("parser_version") == 6:
                    log(sid + ": скачанный оригинал совпадает с проверенным снимком; используем кэш OCR")
                    doc = {**snapshot, "url": final_url, "updated_at": now()}
                else:
                    title, blocks = parse_pdf(raw, source, log) if source["kind"] == "anre_pdf" else parse_html(raw)
                    doc = {"id": sid, "source": "ANRE" if sid.startswith("anre") else "Rapid SCADA 6.4",
                           "url": final_url, "title": title, "language": source.get("language", "ru"), "note": source.get("note", ""),
                           "sha256": digest(raw), "updated_at": now(), "blocks": blocks, "parser_version": 6}
                # Keep last good processed document if downloading/parsing fails.
                raw_path.write_bytes(raw)
                write_json(output, doc)
                status = "updated" if old else "new"
                old = {"url": url, "sha256": digest(raw), "etag": headers.get("ETag"),
                       "last_modified": headers.get("Last-Modified"), "parser_version": 6}
            old.update({"checked_at": now(), "last_error": None})
            state[sid] = old
            item = {"id": sid, "status": status, "at": now()}
        except Exception as exc:
            item = {"id": sid, "status": "error", "error": str(exc), "at": now()}
            old.update({"last_error": str(exc), "checked_at": now()})
            state[sid] = old
        results.append(item)
        write_json(DATA / "state.json", state)
        with (DATA / "grabber_log.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        log(f"{sid}: {item['status']} {item.get('error', '')}")
        time.sleep(0.2)
    return results
