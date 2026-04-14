from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert
from datetime import datetime
from db.dependency import get_db
from model.directory import Directory
from model.image import Image
from logger.logger import logger


"""
    This method is used for inserting ALL directories that contain images.
"""


def insert_all_dirs_and_images(path: Path, db: Session, batch_size: int = 100):
    num_of_inserted_images = 0
    num_of_inserted_dirs = 0
    batch = []

    for root, dirs, files in path.walk():
        image_files = [f for f in files if Path(f).suffix.lower() == ".jpg"]

        if not image_files:
            continue

        directory = db.query(Directory).filter(Directory.path == str(root)).first()

        if directory is None:
            logger.debug(f"Inserting new dir - {root}")
            try:
                directory = Directory(path=str(root))
                db.add(directory)
                db.commit()
                db.refresh(directory)
                logger.info(f"Successfully inserted dir - {directory.path}")
                num_of_inserted_dirs += 1
            except Exception as e:
                logger.warning(
                    f"There was an error while adding new directory - {str(e)}"
                )
                db.rollback()
                continue

        for file in image_files:
            batch.append({"name": str(root / file), "directory_id": directory.id})

            if len(batch) == batch_size:
                try:
                    stmnt = insert(Image).values(batch)
                    stmnt = stmnt.on_conflict_do_nothing(constraint="image_unique")
                    result = db.execute(stmnt)
                    db.commit()
                    num_of_inserted_images += result.rowcount
                    batch.clear()

                except Exception as e:
                    db.rollback()
                    logger.warning(
                        f"There was an error while adding batch of images - {str(e)}"
                    )

        if batch:
            try:
                stmnt = insert(Image).values(batch)
                stmnt = stmnt.on_conflict_do_nothing(constraint="image_unique")
                result = db.execute(stmnt)
                db.commit()
                num_of_inserted_images += result.rowcount
                batch.clear()
            except Exception as e:
                db.rollback()
                logger.warning(
                    f"There was an error while adding batch of images - {str(e)}"
                )

    logger.info(
        f"Inserted {num_of_inserted_dirs} new directories and {num_of_inserted_images} new images"
    )


def traverse_and_persist_directories(
    root: Path, db: Session, batch_size: int = 100
) -> tuple[int]:
    num_of_inserted_dirs = 0
    num_of_inserted_images = 0

    for x in root.iterdir():
        if x.is_dir() and x.name.isdecimal():
            result = traverse_and_persist_directories(x, db, batch_size)
            num_of_inserted_dirs += result[0]
            num_of_inserted_images += result[1]

        elif x.is_dir():
            directory = db.query(Directory).filter(Directory.path == str(x)).first()
            if directory is None:
                logger.debug(f"Inserting new dir - {str(x)}")
                try:
                    directory = Directory(path=str(x))
                    db.add(directory)
                    db.commit()
                    db.refresh(directory)
                    logger.info(f"Successfully inserted dir - {directory.path}")
                    num_of_inserted_dirs += 1
                    num_of_inserted_images += collect_and_insert_images(
                        dir=x, dir_id=directory.id, batch_size=batch_size, db=db
                    )
                except Exception as e:
                    logger.warning(
                        f"There was an error while adding new directory - {str(e)}"
                    )
                    db.rollback()
                    continue
            else:
                num_of_inserted_images += bulk_insert_images(
                    dir=x, dir_id=directory.id, batch_size=batch_size, db=db
                )

    return (num_of_inserted_dirs, num_of_inserted_images)


def collect_and_insert_images(
    dir: Path, dir_id: int, batch_size: int, db: Session
) -> int:

    num_of_inserted_images = 0

    logger.debug(f"Inserting images for directory - {str(dir)}")

    for dirpath, dirnames, filenames in dir.walk():
        image_files = [f for f in filenames if Path(f).suffix.lower() == ".jpg"]

        num_of_inserted_images += bulk_insert_images(
            dirpath, dir_id, image_files, batch_size, db
        )

    logger.info(f"Inserted {num_of_inserted_images} new images")

    return num_of_inserted_images


def bulk_insert_images(
    dir: Path, dir_id: int, image_files: list[str | Path], batch_size: int, db: Session
) -> int:

    batch = []
    num_of_inserted_images = 0
    for file in image_files:
        batch.append({"name": str(dir / file), "directory_id": dir_id})

        if len(batch) == batch_size:
            try:
                stmnt = insert(Image).values(batch)
                stmnt = stmnt.on_conflict_do_nothing(constraint="image_unique")
                result = db.execute(stmnt)
                db.commit()
                num_of_inserted_images += result.rowcount
                batch.clear()
            except Exception as e:
                db.rollback()
                logger.warning(
                    f"There was an error while adding batch of images - {str(e)}"
                )

    if batch:
        try:
            stmnt = insert(Image).values(batch)
            stmnt = stmnt.on_conflict_do_nothing(constraint="image_unique")
            result = db.execute(stmnt)
            db.commit()
            num_of_inserted_images += result.rowcount
            batch.clear()
        except Exception as e:
            db.rollback()
            logger.warning(
                f"There was an error while adding batch of images - {str(e)}"
            )

    return num_of_inserted_images


if __name__ == "__main__":
    with get_db() as db:
        now = datetime.now()
        logger.debug(f"Beginig of insertion - {now.isoformat()}")

        result = traverse_and_persist_directories(
            Path(""), db
        )

        logger.info(f"Inserted {result[0]} new directories and {result[1]} new images")

        logger.debug(f"Insertion finished. Time it took - {datetime.now() - now}")
