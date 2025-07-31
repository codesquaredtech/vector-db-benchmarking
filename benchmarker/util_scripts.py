import numpy as np
import pandas as pd
from app.logger import logger

INPUT_FILE_PATHS = [
    "./input/embeddings_dino_Krstenje - 21. jun 2020. Bogdan_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Krstenje - 8. oktobar 2020. - Krtolica - Indjija_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 11. Januar 2020. - Nikola_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 11. Oktobar 2020. - Jelena i Stefan Bo Inside_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 11. Septembar 2020. - Alaska Terasa - Mirjana i Aleksandar_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 11. septembar 2020. - Jelena i Damir - Greenday_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 12. jun 2021. - Ivana, Vidikovac_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 12. Septembar 2020. - Jasmina i Bojan - Kristal_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 12. Septembar 2020. - Jelena i Milan - Piknik_2025-04-05_22-18-03.parquet",
    "./input/embeddings_dino_Svadba - 12. Septembar 2020. - Jelena i Srdjan - restoran Dunav_2025-04-06_10-39-26.parquet",
    "./input/embeddings_dino_Svadba - 13. jun 2020. - Malinovicevi_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 13. Septembar 2020. - Katarina i Veljko - Zal za mladost_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 13. Septembar 2020. - Kovilj_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 13. septembar 2020. - Marija i Darko - Vrdnicka kula_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 14. Novembar 2020. - Zeljka i Nebojsa_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 15. maj 2021. Sandra i Dragan_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 15. Novembar 2020. - Danijel i Dijana_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 16. Oktobar 2020. - Tamara i Sinisa - GreenDay_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 17. April 2021. - Brana i Nemanja_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 17. Oktobar 2020. - Nina i Nenad - RiverSide pool_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 18. oktobar 2020. - Ana Marija i Juda - Vidikovac_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 2. februar 2020. - Milica i Jovan - Vidikovac_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 2. Oktobar 2020. - Maja (marijana) i Bojan_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 20. septembar 2020. Ubovic_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 21. Avgust 2021. - Natasa i Predrag, Marina_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 21. Avgust 2021. - Nevena i Nemanja, Sombor_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 21. Jun 2021. - Reset, Jelena i_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 21. maj 2021. Tamara i Nikola - salac Bulac_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 22. Novembar 2020. - Jovana i Vladimir_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 23. Januar 2021. - Teodora i D_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 23. Oktobar 2021. - Jovana i Milos_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 24. Septembar 2021. - Dijana i Milos_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 25. April 2020.  - Tijana i Petar - samo maticar_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 25. Jun 2021. - Teodora i Nemanja Jovic - Vidikovac_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 25. Septembar 2020. - Kesten_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 25. septembar 2021. - Jelena i Dusan - Fontana, B. Palanka_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 25. Septembar 2021. - Stasa i Igor, Alaska Barka_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 26. jul 2020. - Marko Marinkovic_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 26. jul 2020. - Slobodanka i Ilija_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 26. jun 2021. - Zorana i Nikola_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 26. septembar 2021. - Milana i Sretko, Kum_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 26. septembar 2021. - Milica i Aleksandar - Alaska barka_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 27. jun 2020. - Eksluziv - Teodora i Bojan_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 27. Jun 2021. - Gordana i Momir - Zal za Mladost_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 28. Avgust 2021. - Marijana i Milos, Subotica_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 28. jun 2020. - Duska i Marko - Kum_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 28. jun 2020. - Vidikovac - Sandra i Miroslav_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 28. jun 2021. - Jovana i Vladimir_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 29. Avgust 2020 - Bojana i Marko - Sasin Salas_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 29. avgust 2020. - Iva i Mihailo_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 29. avgust 2020. - Zal za Mladost - Ana i Aleksandar_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 29. maj 2021. Jelena i Vladan - B. Petrovac_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 3. jul 2020. - Sanja i Nemanja - Alaska Barka_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 3. Jul 2021. - Bojana i Mladen_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 3. oktobar 2020. - Aleksandar - Alaska Barka_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 3. Oktobar 2020. - Tijana i Zarko_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 31. Maj 2020. - Kostresevic_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 4. jun 2020. - Sonja i Milos - yellow house_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 4. Oktobar 2020. - Katarina i Srdjan - Ada_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 5. jun 2021. - Andjela, West Exit_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 5. Septembar 2020. - Dragana i Nikola_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 6. jun 2020. - Beskraj_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 6. Septembar 2020. - Bojana_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadbe - 16. maj 2020. - Biljana i Damir - Session_2025-04-06_12-01-18.parquet",
]


def retrieve_embeddings_from_parquet_files(file_paths):
    embeddings_list = []
    for file_path in file_paths:
        try:
            embeddings = pd.read_parquet(file_path, engine="pyarrow")
            embeddings_list.append(embeddings)
        except Exception as e:
            logger.error(
                f"An error occurred while retrieving embeddings from {file_path}: {e}"
            )

    all_embeddings = pd.concat(embeddings_list, ignore_index=True)
    return all_embeddings


def save_embeddings_to_binary(all_embeddings, vector_column_name, output_path):
    try:
        vectors = np.vstack(all_embeddings[vector_column_name].values)
    except Exception as e:
        logger.error(f"Error stacking embeddings: {e}")
        return

    print(f"Shape: {vectors.shape}")
    print(f"Memory size: {vectors.nbytes / (1024**2):.2f} MB")

    vectors.astype(np.float32).tofile(output_path)


def measure_vectors_size():
    embeddings = retrieve_embeddings_from_parquet_files(INPUT_FILE_PATHS)
    save_embeddings_to_binary(
        embeddings, "embedding", "./results/embeddings_medium_dino.bin"
    )


if __name__ == "__main__":
    measure_vectors_size()
