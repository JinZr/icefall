import argparse
import logging
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from pandas import read_excel

# TARGET_EVENTS = ["低通气", "A. 阻塞性", "A. 中枢性", "A. 混合性"]
LABEL_MAPPING = {
    "正常": 0,
    "低通气": 1,
    "A. 阻塞性": 2,
    "A. 中枢性": 3,
    "A. 混合性": 4,
    "单一打呼": 5,
    "鼾声串": 5,
}
TARGET_EVENTS = LABEL_MAPPING.keys()


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-xls", type=str, required=True, help="Path to the xls file."
    )
    parser.add_argument(
        "--output-csv", type=str, required=True, help="Path to the output csv file."
    )
    parser.add_argument("--edf-date-dir", type=str, default="./edf_date")

    return parser


def read_time(date_file: str):
    with open(date_file) as fin:
        line = fin.readlines()
    assert len(line) == 1, line
    return datetime.strptime(line[0], "%Y-%m-%d %H:%M:%S.%f")


def get_rows(xls_file):
    df = read_excel(xls_file)
    return df.values


def filter_rows(rows):
    return [row for row in rows if row[-1] in TARGET_EVENTS]


def to_list(rows):
    if type(rows) == list:
        return [row.tolist() for row in rows]
    else:
        return rows.tolist()


def append_relative_timestamp(start_time, rows):
    return [
        [
            row[0],  # Absolute timestamp
            row[0] - start_time,  # Relative timestamp
            round(float(row[1]), ndigits=2),  # Duration
            row[2],  # Event
        ]
        for row in rows
    ]


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_xls = Path(args.input_xls)
    output_csv = Path(args.output_csv)
    edf_date_dir = Path(args.edf_date_dir)

    assert edf_date_dir.exists(), f"{edf_date_dir} does not exist"
    start_time_path = edf_date_dir / f"{input_xls.stem}.txt"
    start_time = read_time(start_time_path)

    rows = get_rows(input_xls)
    first_row, last_row = rows[1], rows[-1]
    first_row = to_list(first_row)
    last_row = to_list(last_row)
    # The first row is the header processed automatically by pandas
    # The second row is ``[]	[秒]	[]`` which is not useful

    rows = filter_rows(rows)
    rows = to_list(rows)
    rows = append_relative_timestamp(start_time, rows)

    with open(output_csv, "w") as fout:
        for row in rows:
            fout.write(",".join([str(x) for x in row]) + "\n")

    logging.info(f"==> Info for {input_xls.name}:")
    logging.info(f"Number of rows: {len(rows)}")
    logging.info(f"Total duration of sleep: {last_row[0] - start_time}s")
    for event in TARGET_EVENTS:
        logging.info(f"Name: {event}")
        logging.info(f"Number: {len([row for row in rows if row[-1] == event])}")
        if len([row for row in rows if row[-1] == event]) > 0:
            logging.info(
                f"Average duration: {mean([row[2] for row in rows if row[-1] == event])}s"
            )
            try:
                logging.info(
                    f"Standard deviation: {stdev([row[2] for row in rows if row[-1] == event])}s"
                )
            except:
                logging.error("Standard deviation is not available.")
            logging.info(f"Max: {max([row[2] for row in rows if row[-1] == event])}s")
            logging.info(f"Min: {min([row[2] for row in rows if row[-1] == event])}s")
            logging.info(
                f"Total duration: {sum([row[2] for row in rows if row[-1] == event])}s"
            )
