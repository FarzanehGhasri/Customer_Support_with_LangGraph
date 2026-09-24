# راهنمای اجرا — Running the project

> خلاصه: بدون کلید API هم کل سیستم اجرا می‌شود.
> Short version: the whole system runs without an API key.

---

## پیش‌نیاز

Python **3.10** یا بالاتر.

```bash
python3 --version
```

---

## گام ۱ — گرفتن کد

```bash
git clone https://github.com/FarzanehGhasri/Customer_Support_with_LangGraph.git
cd Customer_Support_with_LangGraph
git checkout claude/awesome-brahmagupta-4bgqv5
```

اگر کد را از قبل دارید:

```bash
git fetch origin
git checkout claude/awesome-brahmagupta-4bgqv5
git pull
```

---

## گام ۲ — محیط مجازی و نصب پکیج‌ها

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

حدود ۳۰ ثانیه طول می‌کشد.

---

## گام ۳ — تست (بدون کلید API)

```bash
pytest -q
```

انتظار: `193 passed`.
اگر اینجا سبز شد، یعنی نصب کاملاً درست است.

```bash
pytest -q tests/test_spec_compliance.py   # ۳۵ تست، بند به بند مطابق PDF
```

---

## گام ۴ — اجرای سه سناریو در ترمینال

```bash
python scripts/demo.py --offline
```

سه سناریوی خواسته‌شده در PDF را پشت سر هم اجرا می‌کند. بدون کلید، بدون هزینه.

گزینه‌های دیگر:

```bash
python scripts/demo.py --offline --scenario 2     # فقط یک سناریو
python scripts/demo.py --offline --verbose        # نمایش تصمیم هر گره
python scripts/demo.py --offline --dynamic        # با graph.interrupt() به‌جای update_state
```

---

## گام ۴.۵ — گفتگوی تعاملی (پیام خودتان را بنویسید)

```bash
python scripts/chat.py --offline
```

یک REPL باز می‌شود و می‌توانید هر پیامی بنویسید. هر پیام در **همان مکالمه** می‌ماند،
پس تاریخچه حفظ می‌شود.

زیر هر جواب یک خط خاکستری می‌بینید که نشان می‌دهد چه اتفاقی افتاده:

```
bot> ## Reset a forgotten password ...
     [Technical | Neutral | tools: search_knowledge_base(...) -> grounded (best=0.63, ...)]
```

### دستورهای داخل گفتگو

| دستور | کار |
|---|---|
| `/help` | فهرست دستورها و نمونه پیام‌ها |
| `/new` | شروع مکالمهٔ تازه (پاک کردن تاریخچه) |
| `/user <id>` | تغییر شناسهٔ حساب (`12345` منقضی، `67890` فعال، `11111` آزمایشی، `22222` لغو شده) |
| `/state` | نمایش کامل state گراف |
| `/history` | نمایش رونوشت مکالمه |
| `/tools` | فهرست ابزارها |
| `/kb <query>` | جستجوی مستقیم در knowledge base، بدون دخالت ایجنت‌ها |
| `/quit` | خروج |

### وقتی ربات شناسه لازم دارد

اگر سوالی بپرسید که به شناسهٔ حساب نیاز دارد و شناسه را نداده باشید، ربات **می‌پرسد**،
بعد شناسه را در فایل مشتریان جستجو می‌کند و بر اساس نتیجه جواب می‌دهد:

```
you> My subscription is not working
bot> Of course -- could you tell me your account ID so I can look that up?
     [Billing | Neutral | tools: none]

you> 99999
bot> I could not find any account with the ID '99999'. Could you double-check it?
     [Billing | Neutral | tools: check_subscription_status(99999) -> not found]

you> 12345
bot> User '12345' is on the 'Pro Monthly' plan; status: expired...
     [Billing | Neutral | tools: check_subscription_status(12345) -> expired]
```

همین برای بازگشت وجه هم هست: «I want a refund» → می‌پرسد شمارهٔ تراکنش چیست →
`TXN-1001` تایید، `TXN-1002` طبق قانون رد، `TXN-9999` پیدا نشد.

نکته‌ها:

* شناسهٔ پیدا شده تا آخر مکالمه به خاطر سپرده می‌شود.
* اگر دو بار جواب ندهید، ربات دست از پرسیدن برمی‌دارد.
* با `/user 12345` می‌توانید شناسه را از اول ست کنید تا اصلاً نپرسد.
* شناسه‌های موجود: `12345` منقضی، `67890` فعال، `11111` آزمایشی، `22222` لغو شده.

### وقتی گاردریل فعال می‌شود

اگر پیام عصبانی بنویسید، گراف **متوقف** می‌شود و prompt به `manager>` تغییر می‌کند.
هر چه آنجا بنویسید به‌عنوان پاسخ مدیر ثبت می‌شود و مکالمه ادامه پیدا می‌کند.

```
you> You stole my money! I want a manager
*** The guardrail detected an angry customer, so the graph has HALTED ... ***
manager> I am the senior manager, I will follow up personally.
bot> I am the senior manager, I will follow up personally.
```

گزینه‌های دیگر: `--dynamic` (استفاده از `graph.interrupt()`)، `--verbose`
(نمایش تصمیم هر گره)، `--user <id>`، و بدون `--offline` برای استفاده از مدل واقعی.

### تغییر متن پرامپت‌ها

اگر می‌خواهید رفتار ایجنت‌ها را عوض کنید، متن پرامپت‌ها اینجاست:

| فایل | مربوط به |
|---|---|
| `src/support_system/prompts/triage.py` | ایجنت تریاژ |
| `src/support_system/prompts/specialists.py` | مالی، فنی، عمومی |
| `src/support_system/prompts/guardrail.py` | تحلیل احساسات |

پرامپت‌ها عمداً از کلاس ایجنت‌ها جدا نگه داشته شده‌اند تا تغییرشان به منطق مسیریابی
دست نزند. توجه: پرامپت‌ها فقط در حالت **LIVE** اثر دارند؛ در حالت آفلاین
تصمیم‌ها قاعده‌محورند (`infrastructure/classification/` و `infrastructure/planning/`).

---

## گام ۵ — نوت‌بوک (فایل تحویلی)

```bash
jupyter lab notebooks/customer_support_langgraph.ipynb
```

یا اگر `jupyter lab` را ترجیح نمی‌دهید:

```bash
jupyter notebook notebooks/customer_support_langgraph.ipynb
```

خروجی‌های همهٔ سلول‌ها از قبل ذخیره شده‌اند، پس بدون اجرا هم قابل خواندن است.
برای اجرای دوباره: **Kernel → Restart & Run All**.

---

## گام ۶ — تصویر گراف (لازم برای تحویل)

```bash
python scripts/render_graph.py
```

فایل‌های زیر ساخته می‌شوند:

| فایل | توضیح |
|---|---|
| `docs/images/support_graph.png` | **تصویر تحویلی** — با `draw_mermaid_png()` |
| `docs/images/support_graph.mmd` | سورس Mermaid |
| `docs/images/support_graph.txt` | نسخهٔ ASCII |

> ⚠️ ساختن PNG به اینترنت نیاز دارد (از سرویس `mermaid.ink` استفاده می‌کند).
> اگر در ایران با خطا مواجه شدید، VPN را روشن کنید و دوباره اجرا کنید،
> یا سورس `.mmd` را در <https://mermaid.live> بچسبانید و از آنجا PNG بگیرید.

---

## گام ۷ (اختیاری) — اجرا با مدل واقعی

تا اینجا همه‌چیز آفلاین بود. برای استفاده از مدل:

### ۷.۱ ساخت فایل `.env`

فایل `.env` عمداً در گیت نیست. آن را در ریشهٔ پروژه بسازید:

```bash
cp .env.example .env
```

و محتوایش را این‌طور کنید:

```ini
SUPPORT_PROVIDER=openai
OPENAI_API_KEY=<کلید خودتان>
SUPPORT_BASE_URL=https://api.gapgpt.app/v1
SUPPORT_MODEL=gpt-4o-mini
SUPPORT_EMBEDDING_MODEL=text-embedding-3-small
SUPPORT_TEMPERATURE=0
```

> `SUPPORT_BASE_URL` بالا **تأیید نشده** است. آدرس درست را از
> <https://gapgpt.app/platform-v2/docs/quickstart> بردارید.

### ۷.۲ تست اتصال

```bash
python scripts/verify_provider.py
```

سه چیز را به ترتیب چک می‌کند و سر اولین خطا می‌ایستد:

1. کلید خوانده شد؟
2. endpoint جواب می‌دهد؟ (و چه مدل‌هایی دارد)
3. `with_structured_output` کار می‌کند؟ ← **مهم‌ترین**

برای دیدن فهرست مدل‌های مجاز:

```bash
python scripts/verify_provider.py --list-models
```

### ۷.۳ اجرا

```bash
python scripts/demo.py          # بدون --offline
```

---

## رفع اشکال

| پیام خطا | علت و راه‌حل |
|---|---|
| `ModuleNotFoundError: support_system` | محیط مجازی فعال نیست → `source .venv/bin/activate` |
| `ModuleNotFoundError: langgraph` | `pip install -r requirements.txt` اجرا نشده |
| `MODE: OFFLINE — no OPENAI_API_KEY` | فایل `.env` ساخته نشده. اگر عمدی است، مشکلی نیست |
| `MODE: OFFLINE — APIConnectionError` | آدرس `SUPPORT_BASE_URL` غلط است یا اینترنت/فیلترینگ |
| `Failed to reach https://mermaid.ink` | فقط روی تصویر گراف اثر دارد → VPN یا mermaid.live |
| `verify_provider` در مرحلهٔ ۳ رد شد | این gateway از JSON schema پشتیبانی نمی‌کند → مدل دیگری امتحان کنید، یا آفلاین اجرا کنید |

---

## تفاوت حالت LIVE و OFFLINE

| | OFFLINE | LIVE |
|---|---|---|
| تریاژ | کلیدواژه‌ای | LLM + `with_structured_output` |
| تحلیل احساسات | کلیدواژه‌ای | LLM |
| جستجوی RAG | TF-IDF | embeddings |
| نگارش پاسخ | متن خام ابزار | LLM |
| هزینه | صفر | چند سنت |
| درستیِ مسیر گراف | ✅ یکسان | ✅ یکسان |

در هر دو حالت، مسیر گراف، ابزارها، توقف روی خشم و HITL **دقیقاً یکی** است؛
فقط جملات در حالت آفلاین خام‌ترند.
