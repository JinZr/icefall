import argparse
import logging
from pathlib import Path
from statistics import mean, stdev

from pandas import read_excel

TARGET_EVENTS = ["低通气", "A. 阻塞性", "A. 中枢性", "A. 混合性"]


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

    return parser


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


def append_relative_timestamp(first_event, rows):
    return [
        [
            row[0],  # Absolute timestamp
            row[0] - first_event[0],  # Relative timestamp
            row[1],  # Duration
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

    rows = get_rows(input_xls)
    first_row, last_row = rows[1], rows[-1]
    first_row = to_list(first_row)
    last_row = to_list(last_row)
    # The first row is the header processed automatically by pandas
    # The second row is ``[]	[秒]	[]`` which is not useful

    rows = filter_rows(rows)
    rows = to_list(rows)
    rows = append_relative_timestamp(first_row, rows)

    with open(output_csv, "w") as fout:
        for row in rows:
            fout.write(",".join([str(x) for x in row]) + "\n")

    logging.info(f"==> Info for {input_xls.name}:")
    logging.info(f"Number of rows: {len(rows)}")
    logging.info(f"Total duration of sleep: {last_row[0] - first_row[0]}s")
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
