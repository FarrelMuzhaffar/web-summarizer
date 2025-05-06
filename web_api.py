from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
import logging
from bs4 import BeautifulSoup
import re
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import signal
import ssl

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi Flask
app = Flask(__name__)

# Konfigurasi CORS dengan logging
CORS(app, resources={r"/*": {
    "origins": ["https://lintasai.com", "https://web-production-a20a.up.railway.app"],
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization"]
}})
logger.info("CORS configured for origins: https://lintasai.com, https://web-production-a20a.up.railway.app")

# Load API Key dari .env
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
logger.info(f"API Key loaded: {'[REDACTED]' if api_key else 'None'}")

# Fungsi validasi URL
def is_valid_url(url):
    pattern = r'^(https?:\/\/)?([\w\-]+(\.[\w\-]+)+[\/]?.*)$'
    return re.match(pattern, url) is not None

# Fungsi ekstrak teks dari website
def extract_text_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        text_elements = soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li"])
        content = " ".join([elem.get_text(strip=True) for elem in text_elements])
        content = re.sub(r'\s+', ' ', content).strip()

        if not content:
            logger.warning("Tidak ada teks yang dapat diekstrak dari URL")
            return None

        return content
    except Exception as e:
        logger.error(f"Gagal mengekstrak teks dari URL: {str(e)}")
        return None

# Tangani sinyal abort untuk logging
def handle_abort_signal(signum, frame):
    logger.error(f"Received signal {signum}, aborting worker")
    raise SystemExit(f"Worker aborted with signal {signum}")

signal.signal(signal.SIGABRT, handle_abort_signal)

# Route untuk menyajikan halaman utama
@app.route('/')
def home():
    logger.info("Serving home page")
    return send_file('web-summarize-ai.html')

# Route untuk menangani preflight OPTIONS secara eksplisit
@app.route('/summarize', methods=['OPTIONS'])
def handle_options():
    logger.info("Handling OPTIONS request for /summarize")
    return jsonify({"status": "ok"}), 200

# Route untuk ringkasan
@app.post('/summarize')
def summarize():
    try:
        logger.info(f"Received {request.method} request to /summarize from {request.origin}")
        data = request.get_json()
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

        # Batasi teks ke 300 kata untuk mempercepat pemrosesan
        if len(content.split()) > 300:
            content = " ".join(content.split()[:300])
            logger.warning("Teks dipotong menjadi 300 kata untuk optimasi")

        if not api_key:
            logger.error("API Key OpenRouter tidak ditemukan di environment!")
            return jsonify({"error": "API Key tidak tersedia"}), 500

        prompt = f"Tolong buatkan ringkasan dalam bentuk poin-poin dan paragraf kesimpulan dari isi website berikut dalam bahasa Indonesia:\n\n{content}"

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

        # Setup retry untuk permintaan OpenRouter
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retries))

        # Tambahkan timeout dan logging waktu
        start_time = time.time()
        try:
            response = session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60  # Timeout 60 detik untuk OpenRouter
            )
            elapsed_time = time.time() - start_time
            logger.info(f"OpenRouter respons dalam {elapsed_time:.2f} detik, status: {response.status_code}")
        except requests.exceptions.Timeout:
            logger.error("Permintaan ke OpenRouter timeout setelah 60 detik")
            return jsonify({"error": "Permintaan ke layanan AI timeout. Coba lagi nanti."}), 504
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Koneksi ke OpenRouter gagal: {str(e)}")
            return jsonify({"error": "Gagal menghubungi layanan AI karena masalah koneksi."}), 503
        except requests.exceptions.SSLError as e:
            logger.error(f"SSL error saat menghubungi OpenRouter: {str(e)}")
            return jsonify({"error": "Gagal menghubungi layanan AI karena masalah SSL."}), 503
        except requests.exceptions.RequestException as e:
            logger.error(f"Gagal menghubungi OpenRouter: {str(e)}")
            return jsonify({"error": f"Gagal menghubungi layanan AI: {str(e)}"}), 503

        if response.status_code != 200:
            logger.error(f"Error dari OpenRouter: {response.status_code} - {response.text}")
            return jsonify({"error": "Gagal meringkas website", "details": response.text}), 500

        try:
            result = response.json()
            summary = result["choices"][0]["message"]["content"]
            logger.info("Successfully generated summary")
            return jsonify({"summary": summary})
        except Exception as e:
            logger.error(f"Gagal parsing JSON dari OpenRouter: {str(e)}")
            return jsonify({"error": f"Gagal membaca respons dari OpenRouter: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Terjadi exception fatal: {str(e)}")
        return jsonify({"error": f"Terjadi error internal: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8000)))