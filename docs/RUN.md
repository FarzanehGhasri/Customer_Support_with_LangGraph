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

فایل `notebooks/customer_support_langgraph.ipynb` **با خروجی‌های ذخیره‌شده** در ریپو
هست. یعنی بعد از `git pull` آماده است و برای خواندن نیازی به اجرا ندارد.

باز کردنش:

```bash
jupyter lab notebooks/customer_support_langgraph.ipynb
# یا
jupyter notebook notebooks/customer_support_langgraph.ipynb
```

### اجرا در VS Code (گام به گام)

۱. پوشهٔ پروژه را در VS Code باز کنید (`File → Open Folder`).
   دو افزونه لازم است و خود VS Code پیشنهادشان می‌دهد:
   **Python** و **Jupyter**.

۲. فایل `notebooks/customer_support_langgraph.ipynb` را باز کنید.

۳. بالا سمت راست روی **Select Kernel** بزنید →
   **Python Environments** → `.venv` را انتخاب کنید (همانی که در گام ۲ ساختید).

   اگر `.venv` در لیست نبود:
   ```bash
   .venv\Scripts\python.exe -m pip install ipykernel
   ```
   بعد VS Code را دوباره باز کنید.

۴. **Run All** را بزنید (یا `Ctrl+Shift+P` → `Notebook: Run All Cells`).

   ترتیب مهم است: سه سلول اول **Setup** هستند و بقیه فقط به آن‌ها وابسته‌اند.
   بعد از اجرای Setup، هر سلول دیگری را می‌توانید جداگانه دوباره اجرا کنید.

۵. `Ctrl+S` بزنید. خروجی‌ها داخل خود فایل `.ipynb` ذخیره می‌شوند — همان چیزی
   که برای تحویل لازم است.

> اگر `ModuleNotFoundError: support_system` دیدید، یعنی kernel اشتباه انتخاب شده
> (مثلاً پایتون سراسری به‌جای `.venv`). دوباره گام ۳ را انجام دهید.

فایل `.vscode/settings.json` از قبل در ریپو هست و مسیر `src` را به Pylance
معرفی می‌کند، پس import‌ها زیرشان خط قرمز نمی‌خورد.

### تصویر گراف داخل نوت‌بوک

سلول بخش ۴ با `draw_mermaid_png()` تصویر را می‌سازد، در
`docs/images/support_graph.png` ذخیره می‌کند، و **داخل خروجی سلول نمایش می‌دهد** —
یعنی تصویر در خود فایل `.ipynb` جاسازی می‌شود.

این سلول به اینترنت نیاز دارد (سرویس `mermaid.ink`). اگر شکست بخورد:

* اگر قبلاً PNG ساخته شده باشد، همان را نشان می‌دهد؛
* وگرنه پیام می‌دهد که `python scripts/render_graph.py` را با VPN اجرا کنید،
  یا محتوای `docs/images/support_graph.mmd` را در <https://mermaid.live> بچسبانید.

### ساختن دوبارهٔ خروجی‌ها

اگر کد را تغییر دادید و می‌خواهید خروجی‌های نوت‌بوک به‌روز شوند:

```bash
python scripts/build_notebook.py     # اجرا و ذخیرهٔ خروجی‌ها
python scripts/make_notebook.py      # ساخت دوبارهٔ خود نوت‌بوک از روی کد
python scripts/make_notebook.py --execute   # هر دو با هم
```

نوت‌بوک از روی `scripts/make_notebook.py` **تولید** می‌شود، نه دستی ویرایش.
اگر می‌خواهید سلولی را عوض کنید، آن اسکریپت را ویرایش کنید و دوباره بسازید —
این‌طور نوت‌بوک هیچ‌وقت از کد عقب نمی‌افتد.

`build_notebook.py` همهٔ سلول‌ها را در یک kernel تمیز و به ترتیب اجرا می‌کند، خروجی‌ها را داخل خود فایل
ذخیره می‌کند، و در آخر بررسی می‌کند که هیچ سلولی خطا نداده باشد:

```
  cells           : 41 (23 code)
  with outputs    : 23/23
  execution errors: 0
  OK: notebooks/customer_support_langgraph.ipynb
```

اگر سلولی خطا بدهد، exit code غیرصفر می‌شود — پس می‌شود در CI هم استفاده کرد.

> چرا اسکریپت به‌جای «فقط Run All»؟ چون تحویلی باید فایلی باشد که خروجی‌ها
> **داخلش** ذخیره شده باشد. اجرای خط‌فرمانی تضمین می‌کند همهٔ سلول‌ها، به ترتیب،
> در یک kernel تازه اجرا شده‌اند.

### گرفتن خروجی HTML یا PDF

```bash
python scripts/build_notebook.py --html      # فایل .html کنار نوت‌بوک
python scripts/build_notebook.py --pdf       # نیاز به نصب LaTeX دارد
```

برای PDF، راه ساده‌تر: HTML بسازید و در مرورگر `Ctrl+P` → Save as PDF.

> فایل‌های `.html` و `.pdf` در `.gitignore` هستند — تحویلی خودِ `.ipynb` است.

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
| `NotJSONError` / `JSONDecodeError` روی نوت‌بوک | فایل `.ipynb` خراب شده — معمولاً کانفلیکت گیت. ↓ بخش زیر |

---

## نوت‌بوک خراب شد (`NotJSONError`)

یک فایل `.ipynb` در واقع یک فایل JSON است. اگر این خطا را دیدید:

```
nbformat.reader.NotJSONError: Notebook does not appear to be JSON
json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes
```

یعنی محتوای فایل خراب شده. **شایع‌ترین علت: کانفلیکت گیت.**

چرا پیش می‌آید؟ نوت‌بوک **همراه با خروجی‌ها** commit شده (چون تحویلی باید خروجی
داشته باشد). وقتی شما محلی اجرایش می‌کنید، خروجی‌ها عوض می‌شوند و فایل dirty می‌شود؛
`git pull` بعدی کانفلیکت می‌دهد و گیت نشانه‌های `<<<<<<<` را داخل فایل می‌گذارد —
که دیگر JSON معتبر نیست.

### تشخیص

```bash
python scripts/build_notebook.py --no-execute
```

خودش دلیل و شمارهٔ خط را می‌گوید و دستور رفع را چاپ می‌کند.

### رفع

```bash
# اگر merge نیمه‌کاره است، اول لغوش کنید
git merge --abort

# نسخهٔ سالم commit‌شده را برگردانید
git checkout -- notebooks/customer_support_langgraph.ipynb

# یا مستقیماً از ریموت، فارغ از وضعیت merge
git fetch origin
git checkout origin/claude/awesome-brahmagupta-4bgqv5 -- notebooks/customer_support_langgraph.ipynb
```

بعدش دوباره:

```bash
python scripts/build_notebook.py
```

### جلوگیری از تکرار

قبل از هر `git pull`، تغییرات محلیِ نوت‌بوک را دور بریزید:

```bash
git checkout -- notebooks/
git pull
```

اگر می‌خواهید اجرای محلی‌تان حفظ شود، اول یک کپی بگیرید:

```bash
cp notebooks/customer_support_langgraph.ipynb notebooks/my_run.ipynb
```

---

## سه حالت اجرا: LIVE / DEGRADED / OFFLINE

| | OFFLINE | DEGRADED | LIVE |
|---|---|---|---|
| تریاژ | کلیدواژه‌ای | کلیدواژه‌ای | LLM + `with_structured_output` |
| تحلیل احساسات | کلیدواژه‌ای | کلیدواژه‌ای | LLM |
| نگارش پاسخ | متن خام ابزار | **LLM** | LLM |
| جستجوی RAG | TF-IDF | TF-IDF | embeddings |
| هزینه | صفر | کم | کم |
| مسیریابی و ابزارها | ✅ درست | ✅ درست | ✅ درست |

در هر سه حالت مسیر گراف، ابزارها، توقف روی خشم و HITL **یکی** است.

### چرا DEGRADED وجود دارد؟

ایجنت تریاژ و گاردریل به `with_structured_output` نیاز دارند. خیلی از
gateway‌های سازگار با OpenAI، چت معمولی را جواب می‌دهند ولی JSON schema /
function calling را **پشتیبانی نمی‌کنند**.

چون همهٔ اجزای این پروژه موقع خطا بی‌صدا fallback می‌کنند، این حالت
**نامرئی** بود: گراف کامل اجرا می‌شد، جواب‌ها روان بودند، ولی *هر* پیام
`General` طبقه‌بندی می‌شد با confidence صفر — هیچ ابزاری اجرا نمی‌شد و مشتری
عصبانی هرگز ارجاع داده نمی‌شد.

حالا دو قابلیت **جداگانه** تست می‌شوند و اگر فقط چت کار کند، برنامه با هشدار
واضح وارد حالت DEGRADED می‌شود:

```
MODE: DEGRADED — openai/gpt-4o-mini
      This endpoint does not support with_structured_output:
      BadRequestError: 'tools' unsupported
      Routing and sentiment use the deterministic classifiers;
      the model still writes the replies.
```

برای بررسی دقیق‌تر gateway خودتان:

```bash
python scripts/verify_provider.py
```
