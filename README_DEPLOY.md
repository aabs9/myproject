# نشر IDS Platform على Render.com

## الملفات المطلوبة (أضفها يدوياً):
- rf_ids_model.pkl
- svm_ids_model.pkl
- scaler.pkl
- label_encoder.pkl

## خطوات النشر:

### 1. GitHub
- أنشئ repository جديد على github.com
- ارفع جميع الملفات بما فيها ملفات .pkl

### 2. Render.com
- اذهب إلى https://render.com
- New → Web Service → ربط GitHub
- Build: pip install -r requirements.txt
- Start: gunicorn app:app
- Plan: Free

### 3. الرابط
ستحصل على رابط مثل: https://ids-platform.onrender.com
