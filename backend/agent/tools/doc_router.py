from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from langchain_core.documents import Document
from langchain_docling.loader import DoclingLoader, ExportType

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ThreadedPdfPipelineOptions,
    EasyOcrOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from docling.datamodel.settings import settings as docling_settings

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc import PictureItem
from transformers import AutoTokenizer
from groq import Groq

GROQ_CLIENT = None
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_TOKENIZER = None


def get_groq_client():
    global GROQ_CLIENT
    if GROQ_CLIENT is None:
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            GROQ_CLIENT = Groq(api_key=api_key, base_url="https://api.groq.com")
            print(f"[vlm] Groq client initialized (api.groq.com)")
    return GROQ_CLIENT


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2"),
            max_tokens=1024,
        )
    return _TOKENIZER


def pil_to_base64(pil_img: Image.Image) -> str:
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def caption_figure_with_groq(
    image_b64: str,
    hint_caption: str = "",
    hint_type: str = "",
) -> str:
    client = get_groq_client()
    if not client or not image_b64:
        return "[Groq VLM unavailable — check GROQ_API_KEY]"

    prompt = (
        "In 2-4 sentences, explain what this figure means and what it demonstrates. "
        "Focus on the finding, trend, or relationship — not visual details like colors, "
        "axis labels, or bar positions. Write as if you are explaining the insight "
        "to someone who cannot see the image. Be concise and direct."
    )
    if hint_caption:
        prompt += f"\nFigure caption: {hint_caption}"

    print(f"[vlm] Sending to Groq VLM: prompt='{prompt[:100]}…'")

    try:
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            max_tokens=350,
            temperature=0.15,
        )
        result = resp.choices[0].message.content.strip()
        print(f"[vlm] Groq response: '{result[:120]}…'")
        return result
    except Exception as e:
        print(f"[vlm] Groq API error: {e}")
        return ""


def detect_mime(filepath: str) -> str:
    try:
        import magic
        return magic.from_file(filepath, mime=True)
    except ImportError:
        import mimetypes
        return mimetypes.guess_type(filepath)[0] or "application/octet-stream"


def is_pure_image(mime: str) -> bool:
    return mime.startswith("image/")


def needs_ocr(filepath: str) -> bool:
    try:
        import pypdf
        reader = pypdf.PdfReader(filepath)
        sample = "".join(page.extract_text() or "" for page in list(reader.pages)[:3])
        return len(sample.strip()) < 50
    except Exception:
        return True


def route_and_parse(
    filepath: str, filename: str, caption_images: bool = True
) -> dict[str, Any]:
    mime = detect_mime(filepath)
    suffix = Path(filepath).suffix.lower().lstrip(".")

    print(f"[doc_router] {filename} | mime={mime} | suffix={suffix}")

    # ── Pure image handling ──────────────────────────────
    if is_pure_image(mime):
        with open(filepath, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        description = (
            caption_figure_with_groq(img_b64) if caption_images else "[Image — captioning disabled]"
        )
        return {
            "filename": filename,
            "parse_method": "groq_vlm_image",
            "page_count": 1,
            "word_count": len(description.split()),
            "total_elements": 1,
            "chunks": [
                Document(
                    page_content=description,
                    metadata={"source": filename, "page_no": 1, "element_type": "picture"},
                )
            ],
            "full_doc": None,
        }

    # ── Document parsing (PDF / DOCX / etc.) ──────────────────────────
    force_ocr = ("pdf" in mime or suffix == "pdf") and needs_ocr(filepath)
    print(f"[doc_router] OCR needed: {force_ocr}")

    import time as _time
    t0 = _time.time()

    # === FAST & CLEAN PIPELINE SETTINGS ===
    docling_settings.debug.profile_pipeline_timings = True
    docling_settings.perf.page_batch_size = 16

    pipeline_opts = ThreadedPdfPipelineOptions(
        do_ocr=force_ocr,
        do_table_structure=True,                    
        generate_picture_images=True,             
        generate_page_images=False,
        images_scale=1.0,
        layout_batch_size=8,
        table_batch_size=4,
        ocr_batch_size=8,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CUDA,
            num_threads=6,
        ),
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_opts,
                backend=PyPdfiumDocumentBackend,
            )
        }
    )
    print(f"[doc_router] Converter created in {_time.time() - t0:.1f}s")

    t1 = _time.time()
    conv_res = converter.convert(filepath)
    doc = conv_res.document
    print(f"[doc_router] Docling convert in {_time.time() - t1:.1f}s")

    t2 = _time.time()
    tokenizer = _get_tokenizer()

    loader = DoclingLoader(
        file_path=filepath,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=tokenizer),
    )
    chunks = loader.load()
    print(f"[doc_router] Chunking in {_time.time() - t2:.1f}s ({len(chunks)} chunks)")
    print(f"[doc_router] Total parse time: {_time.time() - t0:.1f}s")

    # ── VLM captioning for embedded figures (guarded) ─────────────────────────────
    t3 = _time.time()
    if caption_images and get_groq_client():
        picture_map: dict[str, PictureItem] = {}
        for item, _ in doc.iterate_items():
            if isinstance(item, PictureItem):
                picture_map[item.self_ref] = item

        print(f"[doc_router] Found {len(picture_map)} PictureItems")

        if not picture_map:
            print("[doc_router] No pictures detected — skipping VLM captioning")
        else:
            print(f"[doc_router] Starting VLM captioning for {len(picture_map)} figures...")

            # Build page→chunks index
            page_chunks: dict[int, list[int]] = {}
            for ci, chunk in enumerate(chunks):
                meta = chunk.metadata
                dl_meta = meta.get("dl_meta", {})
                for doc_item in dl_meta.get("doc_items", []):
                    provs = doc_item.get("prov", [])
                    for p in provs:
                        pg = p.get("page_no")
                        if pg is not None:
                            page_chunks.setdefault(pg, []).append(ci)

            # Phase 1 & 2: your original VLM logic
            caption_jobs = []
            for ref, pic in picture_map.items():
                try:
                    prov = pic.prov[0] if pic.prov else None
                    page_no = prov.page_no if prov else None
                    if page_no is None or page_no not in page_chunks:
                        continue
                    pil_img = pic.get_image(doc)
                    if not pil_img:
                        continue
                    image_b64 = pil_to_base64(pil_img)
                    hint_cap = pic.caption_text(doc) if hasattr(pic, "caption_text") else ""
                    caption_jobs.append({
                        "ref": ref,
                        "page_no": page_no,
                        "image_b64": image_b64,
                        "hint_cap": hint_cap,
                        "target_idx": page_chunks[page_no][0],
                        "classification": getattr(pic, "classification", ""),
                    })
                except Exception:
                    continue

            print(f"[doc_router]   {len(caption_jobs)} images to caption")

            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _caption_one(job):
                desc = caption_figure_with_groq(
                    job["image_b64"],
                    hint_caption=job["hint_cap"],
                    hint_type=job["classification"],
                )
                return job, desc

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(_caption_one, j): j for j in caption_jobs}
                for future in as_completed(futures):
                    try:
                        job, vlm_desc = future.result()
                        if not vlm_desc:
                            continue
                        target_chunk = chunks[job["target_idx"]]
                        effective_cap = job["hint_cap"]
                        if not effective_cap:
                            import re as _re
                            cap_match = _re.search(r"(Figure\s*\d+[^.]*\.)", target_chunk.page_content[:500])
                            if cap_match:
                                effective_cap = cap_match.group(1).strip()

                        figure_block = f"[Figure Explanation (page {job['page_no']})]: "
                        if effective_cap:
                            figure_block += f"{effective_cap} "
                        figure_block += vlm_desc

                        target_chunk.page_content = f"{figure_block}\n\n{target_chunk.page_content}"
                    except Exception as e:
                        print(f"[doc_router]   ✗ Caption failed: {e}")
    else:
        print("[doc_router] VLM captioning disabled or Groq unavailable")

    print(f"[doc_router] TOTAL time: {_time.time() - t0:.1f}s")

    return {
        "filename": filename,
        "parse_method": f"docling{'_ocr' if force_ocr else ''}",
        "page_count": len(doc.pages) if hasattr(doc, "pages") else 1,
        "word_count": sum(len(c.page_content.split()) for c in chunks),
        "total_elements": len(chunks),
        "chunks": chunks,
        "full_doc": doc,
    }