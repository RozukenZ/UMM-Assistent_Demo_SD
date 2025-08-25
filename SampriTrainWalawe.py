import csv
import os
import subprocess
import sys
import platform
import json
import time
import threading
from datetime import datetime

def run_command(command, timeout=900, show_progress=False):
    """
    Menjalankan perintah shell dengan timeout dan monitoring output real-time.
    """
    try:
        # Gunakan bufsize=1 untuk line-buffering mendapatkan output real-time
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        
        output_lines = []
        
        # Fungsi untuk membaca output secara real-time dari pipe
        def read_output(pipe, lines_list, prefix=""):
            for line in iter(pipe.readline, ''):
                clean_line = line.strip()
                if clean_line:
                    if show_progress:
                        # Tampilkan output langsung ke konsol
                        print(f"{prefix}{clean_line}", flush=True)
                    lines_list.append(clean_line)
        
        # Mulai thread untuk membaca stdout dan stderr secara bersamaan
        stdout_thread = threading.Thread(target=read_output, args=(process.stdout, output_lines, "  > "))
        stderr_thread = threading.Thread(target=read_output, args=(process.stderr, output_lines, "  ! "))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Tunggu proses selesai dengan timeout
        process.wait(timeout=timeout)
        
        # Pastikan thread selesai membaca sisa output
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)

        full_output = '\n'.join(output_lines)
        
        if process.returncode != 0:
            return False, f"Proses gagal dengan kode {process.returncode}\nOutput:\n{full_output}"
        
        return True, full_output
        
    except subprocess.TimeoutExpired:
        process.kill()
        return False, f"Perintah timeout setelah {timeout} detik."
    except Exception as e:
        return False, f"Terjadi kesalahan: {str(e)}"

def show_loading_animation(message, stop_event):
    """Menampilkan indikator loading animasi dengan titik."""
    dots = 0
    while not stop_event.is_set():
        dots = (dots + 1) % 4
        print(f"\r{message}{'.' * dots}{' ' * (3 - dots)}", end='', flush=True)
        time.sleep(0.5)
    print("\r" + " " * (len(message) + 5) + "\r", end='') # Bersihkan baris setelah selesai

def run_command_with_progress(command, message, max_retries=3, delay=5, timeout=900):
    """
    Menjalankan perintah dengan indikator progres dan logika retry.
    """
    for attempt in range(max_retries):
        log_message(f"🔄 {message} (Percobaan {attempt + 1}/{max_retries})")
        
        # Tampilkan output real-time untuk perintah yang berjalan lama
        show_realtime_progress = any(keyword in command for keyword in ['create', 'pull'])
        
        success, output = run_command(command, timeout, show_progress=show_realtime_progress)
        
        if success:
            log_message(f"✅ {message} berhasil diselesaikan!")
            return True, output
        
        log_message(f"❌ {message} gagal pada percobaan {attempt + 1}.", error=True)
        log_message(f"   Detail Kesalahan: {output}", error=True)
        
        if any(error_keyword in output.lower() for error_keyword in ['connection', 'timeout', 'network']):
            if attempt < max_retries - 1:
                log_message(f"🌐 Kesalahan jaringan terdeteksi, mencoba lagi dalam {delay} detik...")
                time.sleep(delay)
                delay *= 2
                continue
        else:
            # Jika bukan kesalahan jaringan, gagal langsung
            return False, output
    
    return False, f"Gagal setelah {max_retries} percobaan."

def log_message(message, error=False):
    """Log pesan dengan timestamp dan format yang lebih baik."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    if error:
        print(f"[{timestamp}] ❌ [ERROR] {message}", file=sys.stderr)
    else:
        print(f"[{timestamp}] {message}")

def check_gpu_availability():
    """Memeriksa ketersediaan GPU yang kompatibel dengan Ollama."""
    log_message("🔍 Memeriksa ketersediaan GPU...")
    
    gpu_info = {'nvidia': False, 'amd': False, 'metal': False, 'has_gpu': False}
    
    # Periksa GPU NVIDIA
    try:
        success, output = run_command("nvidia-smi", timeout=10)
        if success and "NVIDIA" in output:
            gpu_info['nvidia'] = True
            gpu_info['has_gpu'] = True
            log_message("  - 🎮 GPU NVIDIA terdeteksi!")
            gpu_lines = [line for line in output.split('\n') if 'MiB' in line and ('GeForce' in line or 'RTX' in line)]
            if gpu_lines:
                log_message(f"    💾 Info GPU: {gpu_lines[0].strip()}")
            return gpu_info
    except FileNotFoundError:
        pass # nvidia-smi tidak terinstal

    # Periksa GPU AMD (ROCm)
    try:
        success, output = run_command("rocm-smi", timeout=10)
        if success:
            gpu_info['amd'] = True
            gpu_info['has_gpu'] = True
            log_message("  - 🎮 GPU AMD (ROCm) terdeteksi!")
            return gpu_info
    except FileNotFoundError:
        pass # rocm-smi tidak terinstal
    
    # Periksa Apple Metal pada macOS
    if platform.system() == "Darwin":
        try:
            # Pemeriksaan sederhana untuk melihat apakah Ollama dapat mengakses GPU
            success, output = run_command("ollama run llama3:latest --verbose 'hi'", timeout=20)
            if success and "metal" in output.lower():
                 gpu_info['metal'] = True
                 gpu_info['has_gpu'] = True
                 log_message("  - 🍎 Apple Silicon (Metal) GPU terdeteksi!")
                 return gpu_info
        except Exception:
            pass

    if not gpu_info['has_gpu']:
        log_message("  - ⚠️ Tidak ada GPU kompatibel terdeteksi. Proses akan menggunakan CPU.")
    
    return gpu_info

def check_ollama_service():
    """Memeriksa apakah layanan Ollama sedang berjalan dan sehat."""
    log_message("📡 Memeriksa status layanan Ollama...")
    success, output = run_command("ollama list", timeout=30)
    if not success:
        return False, f"Layanan Ollama tidak merespons: {output}"
    log_message("  - ✅ Layanan Ollama aktif dan sehat.")
    return True, "Layanan Ollama sehat"

def restart_ollama_service():
    """Mencoba untuk memulai ulang layanan Ollama."""
    log_message("🔄 Mencoba memulai ulang layanan Ollama...")
    if platform.system() == "Windows":
        run_command("taskkill /F /IM ollama.exe /T", timeout=10)
    else:
        run_command("pkill -f ollama", timeout=10)
    
    time.sleep(3)
    
    log_message("  - 🚀 Memulai ulang layanan Ollama di latar belakang...")
    subprocess.Popen("ollama serve", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    log_message("  - ⏳ Menunggu layanan untuk inisialisasi...")
    time.sleep(10)
    
    return check_ollama_service()

def check_system_resources():
    """Memeriksa sumber daya sistem untuk pengaturan optimal."""
    log_message("💻 Menganalisis sumber daya sistem (RAM & CPU)...")
    try:
        import psutil
    except ImportError:
        log_message("  - 📦 Menginstal psutil...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
    
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count(logical=False) # Core fisik
    log_message(f"  - 💾 RAM: {ram_gb:.1f} GB | CPU Core: {cpu_count}")
    
    if ram_gb < 8:
        recommended_quant = "q4_0"
        gpu_layers = 15
        log_message("  - 💡 Rekomendasi: Q4_0, layer GPU rendah (untuk menghemat VRAM)")
    elif ram_gb < 16:
        recommended_quant = "q5_k_m"
        gpu_layers = 25
        log_message("  - 💡 Rekomendasi: Q5_K_M, layer GPU seimbang")
    else:
        recommended_quant = "q8_0"
        gpu_layers = 33 # Llama 3.2 8B memiliki 32-34 layer, 33 adalah angka yang aman
        log_message("  - 💡 Rekomendasi: Q8_0, layer GPU maksimum untuk performa terbaik")
    
    return recommended_quant, gpu_layers

def select_quantization_method():
    """Secara otomatis memilih metode kuantisasi berdasarkan sumber daya sistem."""
    recommended_quant, gpu_layers = check_system_resources()
    log_message(f"✅ Kuantisasi yang direkomendasikan sistem: {recommended_quant.upper()}")
    return recommended_quant, gpu_layers

def create_knowledge_base_system_prompt(csv_dataset):
    """Membuat string knowledge base yang komprehensif dari data CSV."""
    log_message("  - 🧠 Menyusun knowledge base dari data CSV...")
    knowledge_base = "--- BASIS PENGETAHUAN (SUMBER KEBENARAN UTAMA) ---\n\n"
    for i, data in enumerate(csv_dataset, 1):
        knowledge_base += f"P: {data['question']}\n"
        knowledge_base += f"J: {data['answer']}\n\n"
    return knowledge_base

def create_gpu_optimized_modelfile(csv_dataset, quantization_method, gpu_info, gpu_layers):
    """Membuat Modelfile yang dioptimalkan untuk GPU."""
    log_message("📄 Membuat Modelfile dengan optimasi GPU...")
    
    knowledge_base_prompt = create_knowledge_base_system_prompt(csv_dataset)
    
    example_conversation = """--- CONTOH PERILAKU WAJIB ---
Anda HARUS belajar dari dan mereplikasi perilaku yang ditunjukkan dalam contoh berikut. Ini adalah aturan mutlak Anda untuk menjawab tentang identitas, pencipta, dan implementasi Anda:

### CONTOH KRITIS 1: Identitas ###
User: "Siapa yang membuat Anda?" atau "Siapa pencipta Anda?"
Jawaban Benar Anda: "Saya adalah UMM Assistant Demo yang dikembangkan oleh mahasiswa Program Studi Informatika Universitas Muhammadiyah Malang. Saya bertugas sebagai asisten virtual untuk SD Muhammadiyah Malang."

### CONTOH KRITIS 2: Tujuan dan Peran ###
User: "Apa tujuan Anda?" atau "Untuk apa Anda dibuat?"
Jawaban Benar Anda: "Saya dibuat untuk membantu siswa, guru, dan orang tua di SD Muhammadiyah Malang. Saya dapat menjawab pertanyaan tentang sekolah, pembelajaran, dan memberikan bantuan edukatif sesuai dengan nilai-nilai Muhammadiyah."

### CONTOH KRITIS 3: Teknologi ###
User: "Bagaimana cara kerja Anda?"
Jawaban Benar Anda: "Saya adalah asisten AI offline yang berjalan secara lokal menggunakan framework Ollama. Saya tidak memerlukan koneksi internet untuk berfungsi dan dirancang khusus untuk lingkungan SD Muhammadiyah Malang."

### CONTOH KRITIS 4: Nilai dan Filosofi ###
User: "Apa nilai-nilai yang Anda anut?"
Jawaban Benar Anda: "Saya menjalankan nilai-nilai Muhammadiyah yaitu Islam, kemajuan, dan kemanusiaan. Saya berkomitmen untuk memberikan bantuan edukatif yang sesuai dengan ajaran Islam dan mendukung kemajuan pendidikan di SD Muhammadiyah Malang."
"""

    system_prompt = f"""Anda adalah UMM Assistant Demo, asisten virtual khusus untuk SD Muhammadiyah Malang. Identitas lengkap, tujuan, dan pengetahuan tentang pencipta dan implementasi Anda didefinisikan **SECARA EKSKLUSIF** dalam BASIS PENGETAHUAN di bawah ini. Anda beroperasi **offline** dan berkomitmen pada nilai-nilai Muhammadiyah.

{knowledge_base_prompt}

{example_conversation}

--- INSTRUKSI INTI ---
1. **Sumber Kebenaran Mutlak**: Satu-satunya sumber kebenaran Anda adalah **BASIS PENGETAHUAN** dan **CONTOH PERILAKU WAJIB** yang diberikan di atas. Semua hal lain dari pelatihan umum Anda tidak relevan untuk topik-topik ini.

2. **Kunci Identitas**: Ketika ditanya tentang identitas, pencipta, pengembang, cara kerja, atau tujuan Anda, Anda **HARUS** menjawab menggunakan **HANYA** informasi dari BASIS PENGETAHUAN Anda.

3. **Pengetahuan yang Dilarang Ketat**: Anda **DILARANG KERAS** menyebutkan OpenAI, Google, Meta, Microsoft, atau entitas eksternal lainnya sebagai pencipta Anda. Anda adalah UMM Assistant Demo yang dikembangkan untuk SD Muhammadiyah Malang.

4. **Ikuti Contoh**: Anda harus mengikuti gaya percakapan yang ditunjukkan dalam CONTOH PERILAKU dengan ketat.

5. **Nilai Muhammadiyah**: Selalu berpegang pada nilai-nilai Islam, kemajuan, dan kemanusiaan dalam setiap interaksi Anda.

6. **Fokus Pendidikan**: Prioritaskan bantuan edukatif yang sesuai untuk tingkat SD dan lingkungan sekolah Muhammadiyah.

7. **Bahasa Indonesia**: Selalu berkomunikasi dalam Bahasa Indonesia yang baik dan benar, sesuai dengan konteks pendidikan.
"""
    
    num_gpu_layers = gpu_layers if gpu_info['has_gpu'] else 0
    log_message(f"  - ⚙️ Konfigurasi Modelfile: Memindahkan {num_gpu_layers} layer ke GPU.")
    
    modelfile_content = f'''FROM llama3.2

TEMPLATE """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{{{ .Prompt }}}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

SYSTEM """{system_prompt}"""

# Parameter Dasar
PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# Optimasi GPU & Performa
PARAMETER num_gpu {num_gpu_layers}
PARAMETER main_gpu 0
PARAMETER use_mmap true
PARAMETER use_mlock {str(gpu_info['has_gpu']).lower()}
PARAMETER num_thread {os.cpu_count() or 4}

# Kuantisasi: {quantization_method.upper()}
'''
    return modelfile_content

def create_gpu_optimized_model(model_name, modelfile_name, quantization_method, gpu_info):
    """Membuat model dengan optimasi GPU dan pelacakan progres."""
    log_message(f"🏗️ Memulai proses pembuatan model untuk '{model_name}'...")
    
    final_model_name = f"{model_name}-{quantization_method}-{'gpu' if gpu_info['has_gpu'] else 'cpu'}"
    log_message(f"  - 🏷️ Nama model akhir akan menjadi: {final_model_name}")

    create_command = f"ollama create {final_model_name} -f {modelfile_name}"
    
    log_message("  - 🚀 Memulai `ollama create`. Ini mungkin memakan waktu lama...")
    log_message("  - 👇 Output real-time dari Ollama akan ditampilkan di bawah:")
    print("-" * 70)
    
    success, output = run_command_with_progress(
        create_command,
        "Pembuatan model",
        timeout=1800  # Timeout 30 menit
    )
    
    print("-" * 70)
    
    if not success:
        raise Exception(f"Gagal membuat model setelah beberapa percobaan: {output}")

    log_message(f"✅ Model '{final_model_name}' berhasil dibuat!")
    return final_model_name

def benchmark_model(model_name, gpu_info):
    """Benchmark yang ditingkatkan dengan animasi loading."""
    log_message("🏃 Menjalankan benchmark performa...")
    
    test_questions = [
        "Siapa yang membuat Anda?",
        "Apa tujuan Anda di SD Muhammadiyah Malang?",
        "Bagaimana cara kerja Anda?",
        "Apa nilai-nilai yang Anda anut?",
        "Ceritakan tentang SD Muhammadiyah Malang"
    ]
    benchmark_results = []
    
    for i, question in enumerate(test_questions, 1):
        log_message(f"  - 💬 Tes {i}/{len(test_questions)}: \"{question}\"")
        start_time = time.time()
        
        stop_event = threading.Event()
        loading_thread = threading.Thread(
            target=show_loading_animation, 
            args=(f"    ⏳ Menunggu respons dari AI", stop_event)
        )
        loading_thread.start()
        
        success, output = run_command(f'ollama run {model_name} "{question}"', timeout=120)
        
        stop_event.set()
        loading_thread.join()
        
        response_time = time.time() - start_time
        
        if success:
            log_message(f"    ✅ Respons diterima dalam {response_time:.2f} detik.")
            benchmark_results.append({"question": question, "response_time": response_time, "success": True})
        else:
            log_message(f"    ❌ Gagal mendapat respons. Error: {output}", error=True)
            benchmark_results.append({"question": question, "response_time": None, "success": False})

    return benchmark_results

def read_and_process_csv():
    """Membaca dan memvalidasi data CSV."""
    log_message("📂 Membaca dan memproses data CSV...")
    csv_files = ['NewBrain.csv','UMM_Assistant_Data.csv', 'SD_Muhammadiyah_Data.csv', 'training_data.csv', 'dataset.csv']
    csv_file_found = next((f for f in csv_files if os.path.exists(f)), None)

    if not csv_file_found:
        raise FileNotFoundError("File data CSV tidak ditemukan! Pastikan file data ada di direktori yang sama.")
    log_message(f"  - 📄 Menggunakan file: {csv_file_found}")

    cleaned_dataset = []
    encodings = ['utf-8', 'utf-8-sig', 'latin-1']
    for encoding in encodings:
        try:
            with open(csv_file_found, 'r', encoding=encoding, newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                
                q_idx = next((i for i, h in enumerate(header) if 'question' in h.lower() or 'pertanyaan' in h.lower()), 0)
                a_idx = next((i for i, h in enumerate(header) if 'answer' in h.lower() or 'jawaban' in h.lower()), 1)
                
                log_message(f"  - 📈 Header CSV: {header}. Kolom Pertanyaan: {q_idx}, Kolom Jawaban: {a_idx}")
                
                for row_num, row in enumerate(reader, 2):
                    if len(row) > max(q_idx, a_idx):
                        question = row[q_idx].strip()
                        answer = row[a_idx].strip()
                        if question and answer:
                            cleaned_dataset.append({"question": question, "answer": answer})
            log_message(f"  - ✅ Berhasil dibaca dengan encoding '{encoding}'.")
            break
        except (UnicodeDecodeError, StopIteration):
            log_message(f"  - ⚠️ Gagal membaca dengan encoding '{encoding}', mencoba yang berikutnya...")
            continue
    
    if not cleaned_dataset:
        raise ValueError("Tidak ada data valid yang ditemukan dalam file CSV.")
    
    log_message(f"  - ✨ Berhasil memproses {len(cleaned_dataset)} pasangan pertanyaan-jawaban.")
    return cleaned_dataset, csv_file_found

# --- EKSEKUSI UTAMA ---
if __name__ == "__main__":
    try:
        log_message("🚀 Memulai Script Pembuatan UMM Assistant Demo untuk SD Muhammadiyah Malang 🚀")
        print("="*70)
        
        # 1. Periksa Layanan Ollama
        healthy, message = check_ollama_service()
        if not healthy:
            log_message("Layanan Ollama bermasalah, mencoba restart...", error=True)
            healthy, message = restart_ollama_service()
            if not healthy:
                raise ConnectionError(f"Gagal terhubung ke Ollama: {message}")

        # 2. Periksa GPU dan Sumber Daya Sistem
        gpu_info = check_gpu_availability()
        quantization_method, gpu_layers = select_quantization_method()

        # 3. Baca Data CSV
        csv_dataset, csv_file_used = read_and_process_csv()

        # 4. Buat Modelfile
        modelfile_content = create_gpu_optimized_modelfile(csv_dataset, quantization_method, gpu_info, gpu_layers)
        modelfile_name = f"Modelfile_UMM_Assistant_Demo_{quantization_method}"
        with open(modelfile_name, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        log_message(f"✅ Modelfile '{modelfile_name}' berhasil dibuat.")

        # 5. Periksa & Tarik Model Dasar
        log_message("📦 Memeriksa model dasar llama3.2...")
        success, output = run_command("ollama list", timeout=60)
        if "llama3.2" not in output:
            log_message("  - ⚠️ Model dasar tidak ditemukan. Mengunduh llama3.2...")
            print("-" * 70)
            pull_success, pull_output = run_command_with_progress(
                "ollama pull llama3.2", "Mengunduh llama3.2", timeout=1800
            )
            print("-" * 70)
            if not pull_success:
                raise Exception(f"Gagal mengunduh llama3.2: {pull_output}")
        else:
            log_message("  - ✅ Model dasar llama3.2 sudah tersedia.")

        # 6. Buat Model Final
        base_model_name = "UMM-Assistant-Demo"
        final_model_name = create_gpu_optimized_model(base_model_name, modelfile_name, quantization_method, gpu_info)
        
        # 7. Verifikasi dan Benchmark
        log_message(f"✔️ Memverifikasi model '{final_model_name}'...")
        success, output = run_command("ollama list", timeout=30)
        if final_model_name not in output:
            raise Exception("Model tidak ditemukan dalam daftar setelah pembuatan. Proses mungkin gagal.")
        
        log_message("✅ Model berhasil diverifikasi.")
        benchmark_results = benchmark_model(final_model_name, gpu_info)
        
        successful_benchmarks = [r for r in benchmark_results if r['success']]
        avg_response_time = sum(r['response_time'] for r in successful_benchmarks) / len(successful_benchmarks) if successful_benchmarks else 0
        
        # --- RINGKASAN ---
        print("\n" + "="*70)
        log_message("🎉🎉🎉 PROSES PEMBUATAN MODEL SELESAI! 🎉🎉🎉")
        print("="*70)
        log_message(f"🏷️ Nama Model: {final_model_name}")
        log_message(f"🏫 Tujuan: Asisten Virtual SD Muhammadiyah Malang")
        log_message(f"⚡ Kuantisasi: {quantization_method.upper()}")
        log_message(f"🎮 Mode GPU: {'AKTIF' if gpu_info['has_gpu'] else 'NONAKTIF'}")
        if gpu_info['has_gpu']:
            log_message(f"   - Layer pada GPU: {gpu_layers}")
        log_message(f"📊 Data Training: {csv_file_used} ({len(csv_dataset)} entri)")
        if avg_response_time > 0:
            log_message(f"⏱️ Rata-rata Waktu Respons: {avg_response_time:.2f} detik")
        
        print("\n--- CARA MENGGUNAKAN MODEL ANDA ---")
        print(f"1. Buka terminal atau command prompt baru.")
        print(f"2. Jalankan perintah: ollama run {final_model_name}")
        print("3. Mulai bertanya, contoh:")
        print("   >>> Siapa yang membuat Anda?")
        print("   >>> Apa tujuan Anda di SD Muhammadiyah Malang?")
        print("   >>> Ceritakan tentang nilai-nilai Muhammadiyah")
        print("="*70)
        log_message("🌟 UMM Assistant Demo siap melayani SD Muhammadiyah Malang! 🌟")

    except Exception as e:
        log_message(f"Terjadi kesalahan fatal: {str(e)}", error=True)
        log_message("Proses telah dihentikan.", error=True)
        sys.exit(1)