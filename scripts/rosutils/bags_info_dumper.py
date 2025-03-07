"""
This creates a sqlite db of all the rosbags in a directory. Then it can be used to find rosbags with certain constraints.

Example usage: scripts.rosutils.bags_info_dumper find /path/to/rosbags --db rosbags.db

SQL Query example:
Search for rosbag with a '/tf' topic and a topic finishing by 'imu/mag' but that doesn't have any topic containing 'husky'

WITH BAGS AS (SELECT rosbag_id
              FROM (SELECT rosbag_id -- include
                    FROM topics
                    WHERE topic = '/tf'
                    INTERSECT
                    SELECT rosbag_id
                    FROM topics
                    WHERE topic LIKE '%imu/mag')
              WHERE rosbag_id NOT IN
                    (SELECT rosbag_id -- exclude
                     FROM topics
                     WHERE topic LIKE '%husky%'))
SELECT r.id, -- fetch matching
       r.path,
--        r.hash,
       ROUND(r.duration, 1)                                                            AS duration,
       date(r.start, 'unixepoch')                                                      AS start_date,
--        date(r.end, 'unixepoch')                                                    AS end_date,
       ROUND(r.size / (1024 * 1024), 2)                                                AS size_mb,
       GROUP_CONCAT(t.topic || ' (' || t.type || ', ' || t.message_count || ')', '; ') AS topics_info
FROM rosbag r
         LEFT JOIN topics t ON r.id = t.rosbag_id
WHERE r.id IN (SELECT rosbag_id FROM BAGS)
GROUP BY r.id;

Ideas of improvements:
 - add another subparser to automate sql queries (loss of generality)
 - run multiple process_rosbag in parallel to make it faster
 - add table for tf present in the rosbag
 - add errored bags as start=end=size=-1
"""

import argparse
import hashlib
import logging
import os
import sqlite3
from pathlib import Path

from tqdm import tqdm

import rosbag

logger = logging.getLogger(__name__)


def compute_sha1(filepath: Path, block_size: int = 65536) -> str:
    hasher = hashlib.sha1()
    with filepath.open('rb') as f:
        for block in iter(lambda: f.read(block_size), b""):
            hasher.update(block)
    return hasher.hexdigest()


def init_db(conn: sqlite3.Connection) -> None:
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS rosbag (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            hash TEXT,
            duration REAL,
            start INTEGER,
            end INTEGER,
            size INTEGER
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rosbag_id INTEGER,
            topic TEXT,
            type TEXT,
            message_count INTEGER,
            FOREIGN KEY (rosbag_id) REFERENCES rosbag (id)
        )
    ''')
    conn.commit()


def process_rosbag(conn: sqlite3.Connection, bag_path: Path, compare_method: str) -> None:
    c = conn.cursor()
    if compare_method == "name_only":
        # if name already exists skip
        c.execute("SELECT id FROM rosbag WHERE path LIKE ?", (f"%{bag_path.name}",))
        if c.fetchone():
            tqdm.write(f"Skipping {bag_path.name} based on name.")
            return

    # open bag (reindex if required)
    try:
        bag = rosbag.Bag(str(bag_path), mode="r")
    except rosbag.bag.ROSBagUnindexedException:
        tqdm.write(f"Unindexed bag detected: {bag_path}. Reindexing...")
        try:
            # this is reindexing but not saving the reindex file
            bag = rosbag.Bag(str(bag_path), mode="r", allow_unindexed=True)
            bag.reindex()
            tqdm.write(f"{bag_path}: Reindexation finished.")
        except Exception as e:
            tqdm.write(f"Failed to reindex {bag_path}: {e}")
            return
    except Exception as e:
        tqdm.write(f"Error opening {bag_path}: {e}")
        return

    # get metadata
    info = bag.get_type_and_topic_info()
    topics_info = info.topics if hasattr(info, 'topics') else info

    start_time = bag.get_start_time() if len(topics_info) != 0 else -1
    end_time = bag.get_end_time() if len(topics_info) != 0 else -1
    duration = end_time - start_time
    size = bag_path.stat().st_size

    if compare_method == "rosbag_metadata":
        # if same metadata already exists skip
        c.execute("SELECT id FROM rosbag WHERE start = ? AND end = ? AND size = ?",
                  (int(start_time), int(end_time), size))
        if c.fetchone():
            tqdm.write(f"Skipping {bag_path.name} based on bag metadata.")
            return

    # compute hash and check uniqueness
    file_hash = compute_sha1(bag_path)
    c.execute("SELECT id FROM rosbag WHERE hash = ?", (file_hash,))
    if c.fetchone():
        bag.close()
        tqdm.write(f"Skipping {bag_path.name} based on hash (the file has already been indexed).")
        return

    # insert new data
    c.execute('''
        INSERT OR IGNORE INTO rosbag (path, hash, duration, start, end, size)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (str(bag_path), file_hash, duration, int(start_time), int(end_time), size))
    conn.commit()

    c.execute("SELECT id FROM rosbag WHERE path = ?", (str(bag_path),))
    row = c.fetchone()
    if not row:
        tqdm.write(f"Could not retrieve id for {bag_path}")
        bag.close()
        return
    rosbag_id = row[0]

    for topic, info in topics_info.items():
        c.execute('''
            INSERT INTO topics (rosbag_id, topic, type, message_count)
            VALUES (?, ?, ?, ?)
        ''', (rosbag_id, topic, info.msg_type, info.message_count))
    conn.commit()
    bag.close()


def search_and_save(directory: Path, db_path: Path, include_hidden: bool, compare_method: str) -> None:
    conn = sqlite3.connect(str(db_path))
    init_db(conn)

    try:
        progress = tqdm(directory.rglob("*.bag"))
        for bag_path in progress:
            if not include_hidden and any(part.startswith('.') for part in bag_path.parts):
                continue
            bag_path_short = str(bag_path) if len(str(bag_path)) < 70 else str(bag_path)[-70:]
            progress.set_description("Processing: [...]%s" % bag_path_short)
            process_rosbag(conn, bag_path, compare_method)
    except Exception as e:
        logger.error("Error %s", e)

    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Rosbag DB Utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    find_parser = subparsers.add_parser("find", help="Find rosbag files and store info in SQLite DB")
    find_parser.add_argument("directory", type=Path, help="Directory to search for .bag files")
    find_parser.add_argument("--db",
                             default=Path(os.path.dirname(__file__)) / ".." / ".." / "notes" / "exclude" / "rosbags.db",
                             type=Path,
                             help="SQLite database file path (default: %(default)s)")
    find_parser.add_argument("--include-hidden",
                             default=False,
                             action="store_true", help="Include hidden folders in search")
    find_parser.add_argument("--compare-method",
                             choices=["name_only", "rosbag_metadata", "file_hash"],
                             default="rosbag_metadata",
                             help="Comparison method to determine duplicate files (default: %(default)s)")
    args = parser.parse_args()

    if args.command == "find":
        if args.compare_method != "file_hash":
            logging.info(f"You're using {args.compare_method} comparison method which can lead to some errors when"
                         f" determining identical files")
        search_and_save(args.directory, args.db, args.include_hidden, args.compare_method)


if __name__ == "__main__":
    main()
