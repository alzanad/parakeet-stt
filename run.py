import onnx_asr
import os
import sys

# 1. الإعدادات الأساسية
repo_id = "istupakov/parakeet-tdt-0.6b-v3-onnx"
model_path = "./local_model"
audio_file = "en.wav"
output_dir = "output"

# التحقق من الملف الصوتي فوراً لعدم تداخل الكود
if not os.path.exists(audio_file):
    print(f"⚠️ خطأ: لم يتم العثور على الملف '{audio_file}'.")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

# 2. تحميل النموذج
print(f"⏳ جاري تحميل النموذج...")
model = onnx_asr.load_model(repo_id, path=model_path).with_timestamps()

# 3. استخراج النص
print(f"🎙️ جاري استخراج النص بدقة...")
result = model.recognize(audio_file, channel="mean")

# إعداد مسارات الحفظ
base_name = os.path.splitext(os.path.basename(audio_file))[0]
out_txt = os.path.join(output_dir, f"{base_name}.txt")
out_srt = os.path.join(output_dir, f"{base_name}.srt")
out_vtt = os.path.join(output_dir, f"{base_name}.vtt") # المسار الجديد لملف VTT

# 4. حفظ ملف TXT
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(result.text)

# دالة التنسيق الزمني (تم تحديثها لتدعم VTT أيضاً)
def format_time(seconds, is_vtt=False):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    # تحديد الفاصل الزمني: النقطة لـ VTT والفاصلة لـ SRT
    separator = "." if is_vtt else ","
    return f"{h:02d}:{m:02d}:{s:02d}{separator}{ms:03d}"

# 5. حفظ ملفي SRT و VTT معاً
with open(out_srt, "w", encoding="utf-8") as f_srt, open(out_vtt, "w", encoding="utf-8") as f_vtt:
    # كتابة الترويسة الأساسية الخاصة بصيغة VTT
    f_vtt.write("WEBVTT\n\n")

    counter = 1
    current_sentence = ""
    start_time = -1.0
    t_end = 0.0
    split_marks = ['.', '?', '!', '。', '؟', '\n']

    # المرور على الرموز المستخرجة مباشرة
    for i, token in enumerate(result.tokens):
        token_str = str(token).replace(' ', ' ')
        ts = result.timestamps[i]

        # استخراج توقيت البداية والنهاية للرمز
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            t_start, t_end = ts[0], ts[1]
        else:
            t_start = ts
            if i + 1 < len(result.timestamps):
                next_ts = result.timestamps[i+1]
                t_end = next_ts[0] if isinstance(next_ts, (list, tuple)) else next_ts
            else:
                t_end = t_start + 0.3

        if start_time == -1.0:
            start_time = t_start

        current_sentence += token_str

        # شروط إغلاق سطر الترجمة
        has_punct = any(mark in token_str for mark in split_marks)

        is_long_pause = False
        if i + 1 < len(result.timestamps):
            next_ts = result.timestamps[i+1]
            next_start = next_ts[0] if isinstance(next_ts, (list, tuple)) else next_ts
            is_long_pause = (next_start - t_end) > 1.0

        # الكتابة في الملفات إذا تحقق شرط علامة الترقيم أو السكتة
        if has_punct or is_long_pause:
            clean_text = current_sentence.strip()
            if clean_text:
                # كتابة SRT
                f_srt.write(f"{counter}\n{format_time(start_time)} --> {format_time(t_end)}\n{clean_text}\n\n")
                # كتابة VTT (يُكتفى بالتوقيت والنص)
                f_vtt.write(f"{format_time(start_time, is_vtt=True)} --> {format_time(t_end, is_vtt=True)}\n{clean_text}\n\n")
                counter += 1

            current_sentence = ""
            start_time = -1.0

    # حفظ الجملة الأخيرة المعلقة (إن وجدت)
    clean_text = current_sentence.strip()
    if clean_text:
        if start_time == -1.0: start_time = t_end
        f_srt.write(f"{counter}\n{format_time(start_time)} --> {format_time(t_end)}\n{clean_text}\n\n")
        f_vtt.write(f"{format_time(start_time, is_vtt=True)} --> {format_time(t_end, is_vtt=True)}\n{clean_text}\n\n")

print(f"✅ اكتملت العملية باحترافية! تم إنشاء:\n   📄 {out_txt}\n   🎬 {out_srt}\n   🌐 {out_vtt}")
