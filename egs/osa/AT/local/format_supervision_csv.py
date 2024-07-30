import argparse
import datetime
import logging
from pathlib import Path
from statistics import mean, stdev

from format_supervision_xls import append_relative_timestamp

TARGET_EVENTS = ["低通气", "阻塞性呼吸暂停", "中枢性呼吸暂停", "混合型呼吸暂停"]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input-csv", type=str, required=True, help="Path to the csv file."
    )
    parser.add_argument(
        "--output-csv", type=str, required=True, help="Path to the output csv file."
    )

    return parser


def get_rows(txt_file):
    with open(txt_file, "r") as fin:
        lines = fin.readlines()

    # Skip the first line, which is the header of the csv file
    # 类型,睡眠期,时间,时期,持续时间,体位,验证
    return lines[1:]


def filter_rows(rows):
    return [row for row in rows if row[-1] in TARGET_EVENTS]


def to_list(rows):
    res = []
    if type(rows) == list:
        last_dt = None
        for row in rows:
            split = row.split(",")

            # parsing the time
            hour, minute, second = split[2].split(":")

            curr_dt = datetime.datetime(2024, 1, 1, int(hour), int(minute), int(second))

            if last_dt is not None:
                if curr_dt < last_dt:
                    curr_dt += datetime.timedelta(days=1)

            res.append(
                [
                    curr_dt,  # Absolute timestamp
                    split[4],  # Duration
                    split[0],  # Event
                ]
            )

            last_dt = curr_dt

        return res
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    rows = get_rows(input_csv)

    rows = to_list(rows)
    rows = filter_rows(rows)

    # rows = append_relative_timestamp(first_event, rows)

    # Target CSV format:
    # Absolute timestamp, Relative timestamp, Duration, Event

    with open(output_csv, "w") as fout:
        for row in rows:
            fout.write(",".join([str(x) for x in row]) + "\n")

    logging.info(f"==> Info for {input_csv.name}:")
    logging.info(f"Number of rows: {len(rows)}")
    for event in TARGET_EVENTS:
        logging.info(f"Name: {event}")
        logging.info(f"Number: {len([row for row in rows if row[-1] == event])}")
        if len([row for row in rows if row[-1] == event]) > 0:
            logging.info(
                f"Average duration: {mean([float(row[-2]) for row in rows if row[-1] == event])}s"
            )
            try:
                logging.info(
                    f"Standard deviation: {stdev([float(row[-2]) for row in rows if row[-1] == event])}s"
                )
            except:
                logging.error("Standard deviation is not available.")
            logging.info(
                f"Max: {max([float(row[-2]) for row in rows if row[-1] == event])}s"
            )
            logging.info(
                f"Min: {min([float(row[-2]) for row in rows if row[-1] == event])}s"
            )
            logging.info(
                f"Total duration: {sum([float(row[-2]) for row in rows if row[-1] == event])}s"
            )
