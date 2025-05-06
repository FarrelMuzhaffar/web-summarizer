from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
import logging
from bs4 import BeautifulSoup
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup Flask
app = Flask(__name__)
CORS(app, resources={r"/summarize": {"origins": ["https://lintasai.com"]}})

# Load API Key dari .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
logger.info(f"API Key loaded: {'[REDACTED]' if api_key else 'None'}")  # Log tanpa menampilkan API Key

# Fungsi ekstrak teks dari website
def extract_text_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Hapus elemen yang tidak diinginkan (script, style, dll.)
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Ambil teks dari elemen utama (p, h1-h6, dll.)
        text_elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
        content = " ".join([elem.get_text(strip=True) for elem in text_elements])

        # Bersihkan teks (hapus karakter berulang, spasi berlebih)
        content = re.sub(r'\s+', ' ', content).strip()

        if not content:
            logger.warning("Tidak ada teks yang dapat diekstrak dari URL")
            return None

        return content
    except Exception as e:
        logger.error(f"Gagal mengekstrak teks dari URL: {str(e)}")
        return None

# Fungsi validasi URL
def is_valid_url(url):
    pattern = r'^(https?:\/\/)?([\w\-]+(\.[\w\-]+)+[\/]?.*)$'
    return re.match(pattern, url) is not None

@app.route("/", methods=["GET"])
def home():
    return "Web Summarizer API is running 🚀", 200

@app.route("/summarize", methods=["POST", "OPTIONS"])
def summarize():
    if request.method == "OPTIONS":
        return '', 200

    try:
        data = request.get_json(force=True)
        logger.info(f"Data diterima: {data}")

        if not data or "web_url" not in data:
            logger.error("web_url tidak ditemukan dalam request")
            return jsonify({"error": "Parameter 'web_url' diperlukan"}), 400

        web_url = data["web_url"]
        logger.info(f"web_url: {web_url}")

        if not is_valid_url(web_url):
            logger.error("URL website tidak valid")
            return jsonify({"error": "URL website tidak valid"}), 400

        content = extract_text_from_url(web_url)
        if not content:
            logger.error("Tidak ada teks yang dapat diekstrak dari website ini")
            return jsonify({"error": "Tidak ada teks yang dapat diekstrak dari website ini"}), 400

        logger.info(f"Jumlah kata teks: {len(content.split())}")

        if len(content.split()) > 10000:
            content = " ".join(content.split()[:10000])
            logger.warning("Teks dipotong menjadi 10.000 kata")

        if not api_key:
            logger.error("API Key OpenRouter tidak ditemukan di environment!")
            return jsonify({"error": "API Key tidak tersedia"}), 500

        prompt = f"Tolong buatkan ringkasan dalam bentuk poin-poin dari isi website berikut dalam bahasa Indonesia:\n\n{content}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://lintasai.com/web-summarizer-ai/",
            "X-Title": "Web Summarizer"
        }

        payload = {
            "model": "microsoft/phi-4-reasoning-plus:free",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that summarizes website content."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        logger.info(f"Status code dari OpenRouter: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Error dari OpenRouter: {response.status_code} - {response.text}")
            return jsonify({
                "error": "Gagal meringkas website",
                "details": response.text,
                "status_code": response.status_code
            }), 500

        try:
            result = response.json()
            summary = result["choices"][0]["message"]["content"]
            return jsonify({"summary": summary})
        except Exception as e:
            logger.error(f"Gagal parsing JSON dari OpenRouter: {str(e)}")
            return jsonify({"error": "Gagal membaca respons dari OpenRouter", "details": str(e)}), 500

    except Exception as e:
        logger.error(f"Terjadi exception fatal: {str(e)}")
        return jsonify({"error": "Terjadi error internal", "details": str(e)}), 500

# Jalankan aplikasi Flask di Railway
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Running on port: {port}")
    app.run(host="0.0.0.0", port=port)