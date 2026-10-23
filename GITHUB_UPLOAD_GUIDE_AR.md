# خطوات رفع المشروع على GitHub

## اسم المستودع المقترح
`yemen-hospital-accessibility-ml`

## الأسهل: GitHub Desktop
1. فك الضغط عن حزمة التسليم، وادخل إلى المجلد `yemen-hospital-accessibility-ml`.
2. افتح GitHub Desktop، اختر File → Add Local Repository أو File → New Repository إذا لم يكن المجلد مستودعًا بعد.
3. اختر المجلد الجاهز واحفظ التغييرات، ثم اختر Publish repository واجعل الرؤية Public بعد مراجعة التراخيص.
4. تأكد من ظهور مجلدات `data`, `models`, `docs`, `src`, `reports` بصورة مستقلة في GitHub.

## باستخدام أوامر Git (بعد إنشاء مستودع فارغ على GitHub)
افتح Terminal داخل المجلد الداخلي `yemen-hospital-accessibility-ml`:
```bash
git init
git add .
git commit -m "Publish Yemen hospital accessibility ML project"
git branch -M main
git remote add origin https://github.com/USERNAME/yemen-hospital-accessibility-ml.git
git push -u origin main
```
استبدل `USERNAME` باسم حسابك الحقيقي ولا تستخدمه كما هو. **لا ترفع ملف ZIP وحده داخل GitHub**؛ ارفع الملفات والمجلدات المفكوكة ليستطيع الجمهور فتحها، ويتمكن Kaggle من تنزيل بياناتها عبر Raw URLs.

## تفعيل الموقع
اذهب إلى Repository Settings → Pages → Build and deployment → Deploy from a branch، وحدد `main` و`/docs`، ثم Save. قد يستغرق ظهور رابط الموقع بعض الوقت، واختبره قبل نشره على فيسبوك.

## قبل رفع المستودع العام
اقرأ `docs/PUBLISHING_CHECKLIST_AR.md` و`docs/DATA_SOURCES_AND_LICENSES.md`. لا تُدرج مفاتيح API، كلمات مرور أو بيانات حساسة.

## لماذا لن نُعدّل Kaggle Notebook الآن؟
لأن الدفتر يحتاج **الرابط الفعلي للمستودع بعد نشره**. عندما تنتهي من الرفع سنبدل مسارات البيانات بروابط Github Raw الحقيقية ونتأكد من بصمات الملفات. دفتر المناقشة الحالي يظل سليماً دون تغيير.
