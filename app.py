import os, io, time, base64, logging, requests
from collections import defaultdict
from functools import wraps
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

GOOGLE_VISION_API_KEY = os.environ.get("GOOGLE_VISION_API_KEY", "")
MAX_SIZE = 10 * 1024 * 1024
INAT_URL = "https://api.inaturalist.org/v1/taxa"

_store: dict = defaultdict(list)

def rate_limited(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        now = time.time()
        _store[ip] = [t for t in _store[ip] if t > now - 60]
        if len(_store[ip]) >= 10:
            return jsonify({"error": "Too many requests. Please wait a minute."}), 429
        _store[ip].append(now)
        return f(*args, **kwargs)
    return decorated

def strip_exif(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA","P","CMYK"):
        img = img.convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", exif=b"", quality=85)
    out.seek(0)
    return out.read()

def google_vision(image_bytes):
    if not GOOGLE_VISION_API_KEY:
        raise ValueError("GOOGLE_VISION_API_KEY not set")
    b64 = base64.b64encode(image_bytes).decode()
    payload = {"requests": [{"image": {"content": b64}, "features": [
        {"type": "LABEL_DETECTION", "maxResults": 15},
        {"type": "WEB_DETECTION", "maxResults": 10},
        {"type": "OBJECT_LOCALIZATION", "maxResults": 5}
    ]}]}
    r = requests.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}",
        json=payload, timeout=20
    )
    if not r.ok:
        try:
            detail = r.json().get("error", {})
        except Exception:
            detail = {"raw": r.text[:300]}
        logger.error(f"Vision {r.status_code} detail: {detail}")
    r.raise_for_status()
    data = r.json()
    resp = data.get("responses", [{}])[0]
    labels = resp.get("labelAnnotations", [])
    web = resp.get("webDetection", {})
    web_labels = web.get("bestGuessLabels", [])
    web_entities = web.get("webEntities", [])

    results = []

    # Web best guess labels are most specific (e.g. "Golden Retriever" not just "Dog")
    for w in web_labels:
        lbl = w.get("label", "").strip()
        if lbl:
            results.append({"name": lbl.capitalize(), "score": 0.95, "source": "web_guess"})

    # Web entities often contain species-level names
    for e in web_entities[:5]:
        desc = e.get("description", "").strip()
        score = float(e.get("score", 0.0))
        if desc and score > 0.5:
            if not any(r["name"].lower() == desc.lower() for r in results):
                results.append({"name": desc.capitalize(), "score": score, "source": "web_entity"})

    # Label annotations as fallback
    for l in labels:
        name = l.get("description", "").strip().capitalize()
        score = float(l.get("score", 0.0))
        if name and not any(r["name"].lower() == name.lower() for r in results):
            results.append({"name": name, "score": score, "source": "label"})

    logger.info(f"Google Vision results: {[r['name'] for r in results[:5]]}")
    return results

def search_inat_species(query):
    """Search iNaturalist at species level for specific identification."""
    try:
        # First try species rank
        r = requests.get(INAT_URL, params={
            "q": query,
            "limit": 10,
            "rank": "species,subspecies,variety,genus,family",
            "locale": "en",
            "order_by": "observations_count"
        }, headers={"User-Agent": "JuniorRangerApp/1.0"}, timeout=8)

        if r.status_code != 200:
            return None

        results = r.json().get("results", [])
        if not results:
            # Fallback — try without rank restriction
            r2 = requests.get(INAT_URL, params={
                "q": query, "limit": 3, "locale": "en",
                "order_by": "observations_count"
            }, headers={"User-Agent": "JuniorRangerApp/1.0"}, timeout=8)
            if r2.status_code == 200:
                results = r2.json().get("results", [])

        if not results:
            return None

        taxon = results[0]
        common_name = taxon.get("preferred_common_name", "")
        scientific_name = taxon.get("name", "")
        rank = taxon.get("rank", "")
        observations = taxon.get("observations_count", 0)

        # Prefer common name but always include scientific name
        display_name = common_name.capitalize() if common_name else scientific_name.capitalize()

        logger.info(f"iNat species: {display_name} ({scientific_name}) [{rank}] - {observations} obs")

        return {
            "display_name": display_name,
            "common_name": common_name.capitalize() if common_name else "",
            "scientific_name": scientific_name,
            "rank": rank,
            "observations_count": observations,
            "inat_id": taxon.get("id")
        }

    except Exception as e:
        logger.warning(f"iNat search failed for '{query}': {e}")
        return None

def find_best_species(google_labels):
    """
    Try each Google Vision result against iNaturalist species database.
    Prioritise web_guess and web_entity sources as they are most specific.
    """
    # Sort — web guesses first as they are most specific
    sorted_labels = sorted(google_labels, key=lambda x: (
        0 if x.get("source") == "web_guess" else
        1 if x.get("source") == "web_entity" else 2
    ))

    for item in sorted_labels[:8]:
        name = item["name"]
        logger.info(f"Trying iNat species search for: {name}")
        result = search_inat_species(name)
        if result:
            return result, name

    return None, ""

def get_wiki(name):
    if not name:
        return ""
    try:
        headers = {"User-Agent": "JuniorRangerApp/1.0"}
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{name.replace(' ','_')}",
            headers=headers, timeout=5
        )
        if r.status_code == 200:
            extract = r.json().get("extract", "")
            if extract:
                sentences = extract.split(". ")
                short = ". ".join(sentences[:2])
                return short if short.endswith(".") else short + "."

        sr = requests.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search",
            "srsearch": name, "format": "json", "srlimit": 1
        }, headers=headers, timeout=5)
        if sr.status_code == 200:
            res = sr.json().get("query", {}).get("search", [])
            if res:
                title = res[0]["title"].replace(" ", "_")
                sr2 = requests.get(
                    f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
                    headers=headers, timeout=5
                )
                if sr2.status_code == 200:
                    extract = sr2.json().get("extract", "")
                    if extract:
                        sentences = extract.split(". ")
                        short = ". ".join(sentences[:2])
                        return short if short.endswith(".") else short + "."
    except Exception as e:
        logger.warning(f"Wiki failed for '{name}': {e}")
    return ""

@app.route("/identify", methods=["POST"])
@rate_limited
def identify():
    # Accept either multipart/form-data (field "image") OR JSON {"image_base64": "..."}.
    image_bytes = None
    if "image" in request.files:
        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "Empty filename."}), 400
        image_bytes = file.read()
    else:
        data = request.get_json(silent=True) or {}
        b64 = data.get("image_base64")
        if b64:
            if b64.startswith("data:"):
                b64 = b64.split(",", 1)[-1]
            b64 = "".join(b64.split())  # strip whitespace/newlines
            try:
                image_bytes = base64.b64decode(b64, validate=False)
            except Exception as e:
                logger.error(f"base64 decode failed: {e}")
                return jsonify({"error": "Invalid base64 image."}), 400
            logger.info(f"Decoded base64 image: {len(image_bytes)} bytes")

    if image_bytes is None:
        return jsonify({"error": "No image. Send multipart field 'image' or JSON 'image_base64'."}), 400
    if len(image_bytes) > MAX_SIZE:
        return jsonify({"error": "Image too large. Max 10MB."}), 413

    try:
        clean = strip_exif(image_bytes)
    except Exception as e:
        logger.error(f"EXIF error: {e}")
        return jsonify({"error": "Could not process image."}), 422

    try:
        google_labels = google_vision(clean)
    except ValueError as e:
        return jsonify({"error": str(e)}), 503
    except requests.exceptions.Timeout:
        return jsonify({"error": "Vision service timed out."}), 504
    except requests.exceptions.HTTPError as e:
        logger.error(f"Google Vision error: {e}")
        return jsonify({"error": "Image recognition service error."}), 502
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal error."}), 500

    if not google_labels:
        return jsonify({"top_prediction": "Unknown", "confidence": 0.0, "description": "", "results": []}), 200

    inat, matched_label = find_best_species(google_labels)

    if inat:
        top_name = inat["display_name"]
        scientific = inat["scientific_name"]
        common = inat["common_name"]
        top_conf = next(
            (l["score"] for l in google_labels if l["name"].lower() == matched_label.lower()),
            google_labels[0]["score"]
        )
    else:
        top_name = google_labels[0]["name"]
        scientific = ""
        common = ""
        top_conf = google_labels[0]["score"]

    # Use scientific name for Wikipedia for most accurate description
    desc = get_wiki(scientific if scientific else top_name)

    results = [{"name": l["name"], "confidence": round(l["score"], 4)} for l in google_labels[:5]]

    logger.info(f"Final: {top_name} / {scientific} ({top_conf*100:.1f}%)")

    return jsonify({
        "top_prediction": top_name,
        "confidence": round(top_conf, 4),
        "scientific_name": scientific,
        "common_name": common,
        "description": desc,
        "results": results
    }), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "google_vision": bool(GOOGLE_VISION_API_KEY),
        "inaturalist": "species-level identification",
        "wikipedia": "enabled"
    }), 200

@app.route("/debug", methods=["GET"])
def debug():
    return jsonify({
        "google_vision_key_loaded": bool(GOOGLE_VISION_API_KEY),
        "key_preview": GOOGLE_VISION_API_KEY[:12] + "..." if GOOGLE_VISION_API_KEY else "none"
    }), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Google Vision key: {bool(GOOGLE_VISION_API_KEY)}")
    app.run(host="0.0.0.0", port=port, debug=False)
