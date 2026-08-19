# Clinical.RAG — FastAPI

نفس الـ pipeline اللي في `Version3PM.ipynb` (PDF → TOC-aware sections → chunking → dedup →
hybrid retrieval (BM25 + Vector) → cross-encoder rerank → agentic evaluation (Gemini) →
grounded generation → citation verification) لكن مقسّم كـ FastAPI service بدل خلية واحدة.

## هيكل المشروع

```
Clinical.RAG/
├── app/
│   ├── main.py                 # نقطة الدخول + lifespan (بناء الـ index عند التشغيل)
│   ├── config.py                # الإعدادات (بتتقرأ من .env)
│   ├── api/
│   │   └── routes.py           # /ask  /health  /reindex
│   ├── models/
│   │   └── schemas.py          # Request/Response models
│   └── core/
│       ├── pdf_processing.py   # استخراج PDF + TOC sections   (Cell 5-6)
│       ├── chunking.py         # تقسيم النص لـ chunks          (Cell 8)
│       ├── dedup.py            # إزالة التكرار                 (Cell 10)
│       ├── indexing.py         # Chroma + BM25 + reranker      (Cell 12)
│       ├── retrieval.py        # fusion + rerank                (Cell 14)
│       ├── llm_client.py       # عميل Gemini
│       ├── llm_schemas.py      # schemas الخاصة بـ Gemini        (Cell 19)
│       ├── evaluator.py        # scope/sufficiency check         (Cell 20)
│       ├── generator.py        # توليد الإجابة المسندة           (Cell 21)
│       ├── verifier.py         # تحقق من citations               (Cell 22)
│       ├── refusal.py          # بناء رد الرفض                    (Cell 23)
│       └── pipeline.py         # ClinicalRAGPipeline الكامل        (Cell 24)
├── data/
│   └── PDFs/
│       ├── ESC.pdf              # ضع ملفاتك هنا
│       └── NICE.pdf
├── .env.example
├── requirements.txt
└── README.md
```

ملحوظة: خلية الـ Evaluation/Baseline-vs-Improved comparison (Cells 17-18) خاصة بقياس الأداء
جوه النوتبوك بس، مش جزء من الـ API نفسها — سيبتها بره التطبيق. لو عايزها كـ endpoint منفصل
(زي `/eval`) قولّي وهضيفها.

## التشغيل

```bash
# 1. بيئة افتراضية
python -m venv venv
source venv/bin/activate        # على ويندوز: venv\Scripts\activate

# 2. تثبيت المكتبات
pip install -r requirements.txt

# 3. اعمل نسخة من .env.example باسم .env واملأ GEMINI_API_KEY فيها
cp .env.example .env

# 4. حط ملفات الـ PDF في data/PDFs/ (ESC.pdf, NICE.pdf)

# 5. شغّل السيرفر
uvicorn app.main:app --reload
```

هيبني الفهرسة (embeddings + BM25 + reranker) مرة واحدة عند بدء التشغيل — ده بياخد وقت
(تحميل النماذج + معالجة الـ PDFs)، فأول تشغيل هيكون أبطأ من اللي بعده.

بعد ما يشتغل، روح على: `http://127.0.0.1:8000/docs` لتجربة الـ API تفاعليًا.

## الـ Endpoints

| Method | Path              | الوظيفة                                                   |
|--------|-------------------|-----------------------------------------------------------|
| GET    | `/api/v1/health`  | حالة الفهرسة (جاهزة ولا لأ) وعدد الـ chunks                |
| POST   | `/api/v1/ask`     | السؤال الطبي → إجابة مسندة بأدلة أو رفض منظم                |
| POST   | `/api/v1/reindex` | إعادة بناء الفهرسة في الخلفية (مثلاً بعد إضافة PDF جديد)     |

### مثال طلب `/ask`

```json
POST /api/v1/ask
{
  "question": "Is antibiotic prophylaxis recommended for patients undergoing dental procedures to prevent infective endocarditis?",
  "k": 8
}
```

## ⚠️ أمان

النوتبوك الأصلي كان فيه `GEMINI_API_KEY` مكتوب صريح كـ default value في الكود. المفتاح ده
بقى مكشوف بمجرد ما اترفع الملف، **لازم تلغيه من Google AI Studio وتطلع مفتاح جديد**، وحطه
في `.env` بتاعك بس (ملف `.env` متسجلش في git — أضفه في `.gitignore`).
