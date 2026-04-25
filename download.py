import os
import urllib.request
import urllib.error
import json
import sys

# إعدادات النموذج ومسار الحفظ
repo_id = "istupakov/parakeet-tdt-0.6b-v3-onnx"
base_url = f"https://huggingface.co/{repo_id}/resolve/main"
api_url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
local_dir = "./local_model"

# إنشاء المجلد إذا لم يكن موجوداً
os.makedirs(local_dir, exist_ok=True)

def get_file_size(url):
    """دالة لجلب الحجم الكلي للملف قبل بدء التحميل"""
    try:
        # استخدام طلب HEAD لجلب الحجم بسرعة دون تحميل الملف
        req = urllib.request.Request(url, method='HEAD', headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return int(response.info().get('Content-Length', -1))
    except urllib.error.HTTPError:
        # في حال لم يدعم الخادم طلب HEAD، نستخدم GET العادي كبديل
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return int(response.info().get('Content-Length', -1))
    except Exception as e:
        return -1

def download_file(url, dest_path):
    """دالة للتحميل مع دعم الاستئناف وشريط التقدم"""
    total_size = get_file_size(url)
    local_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0

    # إذا كان الملف مكتملاً
    if total_size != -1 and local_size == total_size:
        print(f"✅ الملف {os.path.basename(dest_path)} مُكتمل التحميل مسبقاً. تم التخطي.")
        return

    headers = {'User-Agent': 'Mozilla/5.0'}
    mode = 'wb'
    downloaded = 0

    # إعداد الاستئناف إذا كان الملف موجوداً جزئياً
    if local_size > 0 and total_size != -1:
        if local_size < total_size:
            headers['Range'] = f'bytes={local_size}-'
            mode = 'ab' # وضع الإضافة للملف (Append)
            downloaded = local_size
            print(f"\n🔄 جاري استئناف تحميل: {os.path.basename(dest_path)} (تم تحميل {local_size / (1024*1024):.2f} من {total_size / (1024*1024):.2f} ميغابايت)")
        else:
            # إذا كان الحجم المحلي أكبر من الخادم (يحدث لخلل ما)، نعيد التحميل
            local_size = 0

    if mode == 'wb':
        size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        print(f"\n⬇️ بدء تحميل: {os.path.basename(dest_path)} (الحجم: {size_mb:.2f} ميغابايت)")

    req_dl = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req_dl) as response:

            # التحقق مما إذا كان الخادم قد قبل طلب الاستئناف (كود 206) أم أرسل الملف من البداية
            if getattr(response, 'status', 200) == 200 and mode == 'ab':
                mode = 'wb'
                downloaded = 0
                print("⚠️ الخادم لم يقبل الاستئناف، جاري إعادة التحميل من البداية...")

            with open(dest_path, mode) as f:
                block_size = 1024 * 16  # 16 كيلوبايت للدفعة

                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)

                    # تحديث شريط التقدم
                    if total_size > 0:
                        percent = int((downloaded / total_size) * 100)
                        bar = ('=' * (percent // 2)).ljust(50)
                        sys.stdout.write(f"\r[{bar}] {percent}%")
                        sys.stdout.flush()

            print(f"\n✅ اكتمل تحميل {os.path.basename(dest_path)} بنجاح!")

    except Exception as e:
        print(f"\n❌ حدث خطأ أثناء تحميل {os.path.basename(dest_path)}: {e}")

print("🔍 جاري جلب قائمة الملفات مباشرة من خوادم Hugging Face...")
try:
    # جلب قائمة الملفات من واجهة برمجة التطبيقات (API)
    req = urllib.request.Request(api_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        tree = json.loads(response.read().decode())

    # استخراج مسارات الملفات فقط (واستبعاد المجلدات)
    files = [item['path'] for item in tree if item['type'] == 'file']
    print(f"📦 تم العثور على {len(files)} ملف/ملفات للتحميل.\n")

    # تحميل كل ملف في القائمة
    for file_name in files:
        file_url = f"{base_url}/{file_name}"
        dest_path = os.path.join(local_dir, file_name)
        download_file(file_url, dest_path)

    print("\n🎉 تمت عملية التحميل لجميع الملفات بنجاح!")
except Exception as e:
    print(f"⚠️ فشل في جلب قائمة الملفات. تفاصيل الخطأ: {e}")
