import os
import json
import base64
import mimetypes
import threading
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'Models')
DEFAULT_MODEL = 'gemma-4-E4B-it-Q4_K_M.gguf'

current_model = None
current_model_name = ""
current_gpu_layers = None

# ─── حالة الإيقاف ─────────────────────────────────────────────────────────────
abort_flags = {}
abort_lock  = threading.Lock()

# ─── File type helpers ─────────────────────────────────────────────────────────
TEXT_EXTENSIONS = {
    '.txt', '.md', '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.htm',
    '.css', '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.sh', '.bash', '.zsh', '.fish', '.c', '.cpp', '.h', '.hpp', '.cs',
    '.java', '.kt', '.swift', '.go', '.rs', '.rb', '.php', '.lua', '.r',
    '.sql', '.xml', '.svg', '.csv', '.tsv', '.log', '.env', '.gitignore',
    '.dockerfile', '.makefile', '.tex', '.rst', '.org',
}

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.ico', '.tiff', '.tif'}
PDF_EXTENSION = {'.pdf'}
OFFICE_EXTENSIONS = {'.docx', '.xlsx', '.pptx', '.doc', '.xls', '.ppt'}
ARCHIVE_EXTENSIONS = {'.zip', '.tar', '.gz', '.rar', '.7z'}

def get_file_category(filename: str, mime_type: str = '') -> str:
    ext = os.path.splitext(filename.lower())[1]
    if ext in TEXT_EXTENSIONS or mime_type.startswith('text/'):
        return 'text'
    if ext in IMAGE_EXTENSIONS or mime_type.startswith('image/'):
        return 'image'
    if ext in PDF_EXTENSION or mime_type == 'application/pdf':
        return 'pdf'
    if ext in OFFICE_EXTENSIONS:
        return 'office'
    if ext in ARCHIVE_EXTENSIONS:
        return 'archive'
    return 'binary'


def extract_text_from_base64_file(name: str, mime: str, b64_data: str) -> str:
    """محاولة استخراج نص من الملفات الثنائية"""
    ext = os.path.splitext(name.lower())[1]
    
    try:
        raw = base64.b64decode(b64_data)
    except Exception:
        return None
    
    # PDF → استخراج نص
    if ext == '.pdf' or mime == 'application/pdf':
        try:
            import io
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(raw))
                pages = []
                for i, page in enumerate(reader.pages[:30]):  # max 30 pages
                    text = page.extract_text() or ''
                    if text.strip():
                        pages.append(f"[صفحة {i+1}]\n{text}")
                return '\n\n'.join(pages) if pages else None
            except ImportError:
                pass
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(raw)) as pdf:
                    pages = []
                    for i, page in enumerate(pdf.pages[:30]):
                        text = page.extract_text() or ''
                        if text.strip():
                            pages.append(f"[صفحة {i+1}]\n{text}")
                return '\n\n'.join(pages) if pages else None
            except ImportError:
                pass
        except Exception as e:
            print(f"PDF extraction error: {e}")
        return None
    
    # Office → استخراج نص
    if ext == '.docx':
        try:
            import io
            import zipfile
            from xml.etree import ElementTree as ET
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                xml = z.read('word/document.xml')
            root = ET.fromstring(xml)
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            texts = [t.text for t in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if t.text]
            return ' '.join(texts) if texts else None
        except Exception as e:
            print(f"DOCX extraction error: {e}")
        return None
    
    if ext in ('.xlsx', '.xls'):
        try:
            import io
            try:
                import openpyxl
                wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
                lines = []
                for ws in wb.worksheets:
                    lines.append(f"[ورقة: {ws.title}]")
                    for row in ws.iter_rows(values_only=True):
                        row_text = '\t'.join(str(c) if c is not None else '' for c in row)
                        if row_text.strip():
                            lines.append(row_text)
                return '\n'.join(lines) if lines else None
            except ImportError:
                pass
        except Exception as e:
            print(f"XLSX extraction error: {e}")
        return None

    # محاولة قراءة كنص UTF-8
    try:
        return raw.decode('utf-8')
    except Exception:
        pass
    try:
        return raw.decode('latin-1')
    except Exception:
        pass
    
    return None


def format_file_for_prompt(file_obj: dict) -> str:
    name    = file_obj.get('name', 'unnamed_file')
    mime    = file_obj.get('type', '')
    content = file_obj.get('content', '')   # نص أو base64
    is_b64  = file_obj.get('is_base64', False)
    category = get_file_category(name, mime)

    # ─ صورة ─
    if category == 'image':
        return f"[صورة مرفقة: {name} — النموذج النصي لا يستطيع رؤية الصور مباشرة، لكن يمكنك وصفها أو طرح أسئلة عنها]"

    # ─ ملف ثنائي بدون base64 ─
    if not is_b64:
        if content.startswith('[binary file'):
            return f"[الملف: {name} — ملف ثنائي، المحتوى غير متاح للقراءة]"
        # نص عادي
        max_chars = 24000
        truncated = ''
        if len(content) > max_chars:
            content   = content[:max_chars]
            truncated = f'\n... [تم اقتطاع الملف — الحجم يتجاوز {max_chars} حرفاً]'
        return (
            f"--- بداية الملف: {name} ---\n"
            f"{content}{truncated}\n"
            f"--- نهاية الملف: {name} ---"
        )

    # ─ base64: حاول استخراج النص ─
    extracted = extract_text_from_base64_file(name, mime, content)
    if extracted and extracted.strip():
        max_chars = 24000
        truncated = ''
        if len(extracted) > max_chars:
            extracted = extracted[:max_chars]
            truncated = f'\n... [تم اقتطاع الملف — الحجم يتجاوز {max_chars} حرفاً]'
        return (
            f"--- بداية الملف: {name} ({category}) ---\n"
            f"{extracted}{truncated}\n"
            f"--- نهاية الملف: {name} ---"
        )

    return f"[الملف: {name} — نوع: {mime or 'غير معروف'} ({category})، لا يمكن استخراج نص منه]"


def build_prompt(user_text: str, file_objects: list, system_prompt: str = '',
                 history: list = None) -> str:
    """
    history = [ {role:'user'|'assistant', content:'...'}, ... ]
    آخر رسالة في history هي رسالة المستخدم الحالية — لا نضيفها مرتين.
    """
    parts = []

    if system_prompt:
        parts.append(f"System: {system_prompt}\n")

    # ── تاريخ المحادثة السابقة ──────────────────────────────────────────────
    if history:
        for msg in history:
            role    = msg.get('role', '')
            content = msg.get('content', '').strip()
            if not content:
                continue
            if role == 'user':
                parts.append(f"User: {content}")
            elif role in ('assistant', 'ai'):
                parts.append(f"AI: {content}")
        parts.append("")   # سطر فراغ قبل الرسالة الجديدة

    # ── الملفات المرفقة بالرسالة الحالية ───────────────────────────────────
    if file_objects:
        parts.append("الملفات المرفقة:\n")
        for fo in file_objects:
            parts.append(format_file_for_prompt(fo))
            parts.append("")
        parts.append("---")
        parts.append("")

    parts.append(f"User: {user_text}")
    parts.append("AI:")

    return "\n".join(parts)


# ─── Model management ──────────────────────────────────────────────────────────
def get_available_models():
    if not os.path.exists(MODELS_DIR):
        return []
    return sorted([f for f in os.listdir(MODELS_DIR) if f.endswith('.gguf')])


def load_model(model_name: str, n_gpu_layers: int = -1, n_ctx: int = 32768):
    global current_model, current_model_name, current_gpu_layers

    if current_model_name == model_name and current_gpu_layers == n_gpu_layers and current_model is not None:
        return current_model

    model_path = os.path.join(MODELS_DIR, model_name)
    print(f"🔄 جاري تبديل وضع المعالجة فوراً: {model_name} (gpu_layers={n_gpu_layers}, ctx={n_ctx})...")

    if current_model:
        del current_model
        current_model = None

    if n_gpu_layers == 0:
        use_flash_attn = False
        print("🚫 تم الانتقال لوضع المعالج بالكامل (CPU 100%)")
    else:
        use_flash_attn = True
        print(f"⚡ تم تفعيل كرت الشاشة (GPU Layers = {n_gpu_layers})")

    current_model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_batch=512,
        flash_attn=use_flash_attn,
        verbose=False,
    )
    current_model_name = model_name
    current_gpu_layers = n_gpu_layers
    print(f"✅ تم تفعيل الإعداد الجديد بنجاح!")
    return current_model


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/api/models', methods=['GET'])
def list_models():
    models  = get_available_models()
    current = current_model_name
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else '')
    return jsonify({"models": models, "current": current, "default": default})


@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    global current_model_name
    data         = request.json or {}
    model_name   = data.get('model') or current_model_name or DEFAULT_MODEL
    n_gpu_layers = int(data.get('n_gpu_layers', -1))
    n_ctx        = int(data.get('n_ctx', 32768))

    try:
        load_model(model_name, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
        return jsonify({"status": "success", "message": f"تم التحديث الفوري: النموذج={model_name}، كرت الشاشة={n_gpu_layers}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """إرجاع استهلاك الرام الحالي للسيرفر والنموذج"""
    import psutil, os as _os
    proc = psutil.Process(_os.getpid())
    ram_mb = proc.memory_info().rss / 1024 / 1024          # رام البروسيس
    sys_total  = psutil.virtual_memory().total / 1024 / 1024
    sys_used   = psutil.virtual_memory().used  / 1024 / 1024
    sys_avail  = psutil.virtual_memory().available / 1024 / 1024
    return jsonify({
        "process_mb": round(ram_mb, 1),
        "sys_used_mb": round(sys_used, 1),
        "sys_total_mb": round(sys_total, 1),
        "sys_avail_mb": round(sys_avail, 1),
        "percent": round(psutil.virtual_memory().percent, 1),
    })


# ─── Chat endpoint — POST ─────────────────────────────────────────────────────
@app.route('/api/chat/stream', methods=['POST'])
def chat_stream_post():
    data = request.json or {}
    prompt_text   = data.get('prompt', '')
    model_name    = data.get('model', '')
    file_objects  = data.get('files', [])
    stream_id     = data.get('stream_id', '')
    max_tokens    = int(data.get('max_tokens', 16384))
    n_gpu_layers  = int(data.get('n_gpu_layers', -1))
    n_ctx         = int(data.get('n_ctx', 32768))
    system_prompt = data.get('system_prompt', '')
    history       = data.get('history', [])   # ← تاريخ المحادثة

    if stream_id:
        with abort_lock:
            abort_flags[stream_id] = False

    if not current_model or current_model_name != model_name or current_gpu_layers != n_gpu_layers:
        try:
            load_model(model_name, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
        except Exception as e:
            return Response(
                f"data: {json.dumps({'error': str(e)})}\n\n",
                mimetype='text/event-stream'
            )

    full_prompt = build_prompt(prompt_text, file_objects, system_prompt, history)

    def generate():
        try:
            stream = current_model(
                full_prompt,
                max_tokens=max_tokens,
                stop=["User:", "\n\nUser:"],
                stream=True,
            )
            for chunk in stream:
                if stream_id:
                    with abort_lock:
                        if abort_flags.get(stream_id, False):
                            abort_flags.pop(stream_id, None)
                            yield f"data: {json.dumps({'aborted': True})}\n\n"
                            return

                text = chunk['choices'][0]['text']
                if text:
                    yield f"data: {json.dumps({'text': text})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if stream_id:
                with abort_lock:
                    abort_flags.pop(stream_id, None)

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# ─── نبقي GET للتوافق القديم (بدون ملفات) ───────────────────────────────────
@app.route('/api/chat/stream', methods=['GET'])
def chat_stream_get():
    prompt_text   = request.args.get('prompt', '')
    model_name    = request.args.get('model', '')
    stream_id     = request.args.get('stream_id', '')
    max_tokens    = int(request.args.get('max_tokens', 16384))
    n_gpu_layers  = int(request.args.get('n_gpu_layers', -1))
    n_ctx         = int(request.args.get('n_ctx', 32768))
    system_prompt = request.args.get('system_prompt', '')

    if stream_id:
        with abort_lock:
            abort_flags[stream_id] = False

    if not current_model or current_model_name != model_name or current_gpu_layers != n_gpu_layers:
        try:
            load_model(model_name, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
        except Exception as e:
            return Response(
                f"data: {json.dumps({'error': str(e)})}\n\n",
                mimetype='text/event-stream'
            )

    full_prompt = build_prompt(prompt_text, [], system_prompt)

    def generate():
        try:
            stream = current_model(
                full_prompt,
                max_tokens=max_tokens,
                stop=["User:", "\n\nUser:"],
                stream=True,
            )
            for chunk in stream:
                if stream_id:
                    with abort_lock:
                        if abort_flags.get(stream_id, False):
                            abort_flags.pop(stream_id, None)
                            yield f"data: {json.dumps({'aborted': True})}\n\n"
                            return
                text = chunk['choices'][0]['text']
                if text:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if stream_id:
                with abort_lock:
                    abort_flags.pop(stream_id, None)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/abort', methods=['POST'])
def abort_stream():
    data      = request.json or {}
    stream_id = data.get('stream_id', '')
    if stream_id:
        with abort_lock:
            abort_flags[stream_id] = True
        return jsonify({"status": "abort_requested"})
    return jsonify({"error": "stream_id مطلوب"}), 400


if __name__ == '__main__':
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    app.run(port=5000, debug=False, threaded=True)