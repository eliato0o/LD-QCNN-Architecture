import os
import shutil
import pandas as pd
from tqdm import tqdm

def organize_existing_png(source_dir, labels_csv_path, base_output_dir):
    print(f"Lecture des labels depuis {labels_csv_path}...")
    df = pd.read_csv(labels_csv_path)
    labels_dict = df.drop_duplicates(subset=['patientId']).set_index('patientId')['Target'].to_dict()

    # Création des dossiers cibles
    dir_normal = os.path.join(base_output_dir, "0_Normal")
    dir_pneumonia = os.path.join(base_output_dir, "1_Pneumonia")

    os.makedirs(dir_normal, exist_ok=True)
    os.makedirs(dir_pneumonia, exist_ok=True)

    # Récupération de la liste des PNG
    png_files = [f for f in os.listdir(source_dir) if f.endswith('.png')]
    print(f"{len(png_files)} images PNG détectées dans '{source_dir}'.")

    count_0 = 0
    count_1 = 0
    skipped = 0

    for file in tqdm(png_files, desc="Déplacement et tri"):
        patient_id = file.replace('.png', '')

        if patient_id not in labels_dict:
            skipped += 1
            continue

        target = labels_dict[patient_id]
        target_folder = dir_pneumonia if target == 1 else dir_normal

        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(target_folder, file)

        # Déplacement du fichier
        shutil.move(src_path, dest_path)

        if target == 1:
            count_1 += 1
        else:
            count_0 += 1

    print("\n--- Résumé du tri ---")
    print(f"Images déplacées vers '0_Normal'    : {count_0}")
    print(f"Images déplacées vers '1_Pneumonia' : {count_1}")
    if skipped > 0:
        print(f"Images ignorées (ID non trouvé)     : {skipped}")

    # Suppression du sous-dossier all_images s'il est vide
    try:
        os.rmdir(source_dir)
        print(f"Dossier source vide '{source_dir}' supprimé avec succès.")
    except OSError:
        pass

if __name__ == "__main__":
    SOURCE_PNG_DIR = "./rsna_pneumonia_data/all_images"
    LABELS_CSV     = "./stage_2_train_labels.csv"
    OUTPUT_DIR     = "./rsna_pneumonia_data"

    organize_existing_png(SOURCE_PNG_DIR, LABELS_CSV, OUTPUT_DIR)