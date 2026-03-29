import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

def check_labels(label_dir, nc):
    bad_files = []
    for root, _, files in os.walk(label_dir):
        for fname in files:
            if not fname.endswith(".txt"):
                continue
            path = os.path.join(root, fname)
            with open(path, "r") as f:
                for line_num, line in enumerate(f, start=1):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except:
                        continue
                    if cls_id >= nc or cls_id < 0:
                        bad_files.append((path, line_num, cls_id))
    return bad_files


if __name__ == "__main__":
    nc = 3  # jumlah kelas sesuai data.yaml
    bad = []
    for split in ["train", "valid", "test"]:
        bad += check_labels(f"{split}/labels", nc)

    if bad:
        print(f"Ditemukan {len(bad)} baris dengan class_id di luar rentang 0–{nc-1}:")
        for path, line, cls in bad[:20]:
            print(f"{path} (baris {line}): class_id={cls}")
        print("\nPeriksa file di atas — ubah class_id yang salah ke 0–2.")
    else:
        print("Semua label sesuai dengan jumlah kelas (0–2).")
