import argparse
import datetime
import logging
from pathlib import Path
from statistics import mean, stdev

from format_supervision_xls import append_relative_timestamp

TARGET_EVENTS = ["低通气", "阻塞性呼吸暂停", "中枢性呼吸暂停", "混合型呼吸暂停", "打鼾", "周期性呼吸", "觉醒"]


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
    parser.add_argument(
        "--speaker-id",
        type=str,
        required=True,
        help="Speaker ID",
    )
    parser.add_argument("--edf-date-dir", type=str, default="./edf_date")

    return parser


def get_rows(txt_file):
    with open(txt_file, "r") as fin:
        lines = fin.readlines()

    # Skip the first line, which is the header of the csv file
    # 类型,睡眠期,时间,时期,持续时间,体位,验证
    return lines[1:]


def filter_rows(rows):
    return [row for row in rows if row[-1] in TARGET_EVENTS]


def read_time(date_file: str):
    with open(date_file) as fin:
        line = fin.readlines()
    assert len(line) == 1, line
    try:
        return datetime.datetime.strptime(line[0].strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.datetime.strptime(line[0].strip(), "%Y-%m-%d %H:%M:%S.%f")


def to_list(rows):
    res = []
    if type(rows) == list:
        last_dt = None
        for row in rows:
            split = row.split(",")

            # parsing the time
            hour, minute, second = split[2].split(":")

            curr_dt = datetime.datetime(1970, 1, 1, int(hour), int(minute), int(second))

            if last_dt is not None:
                if curr_dt < last_dt:
                    curr_dt += datetime.timedelta(days=1)
            else:
                if curr_dt < datetime.datetime(1970, 1, 1, 12, 0, 0):
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

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    speaker_id = args.speaker_id
    edf_date_dir = Path(args.edf_date_dir)

    assert edf_date_dir.exists(), f"{edf_date_dir} does not exist"
    start_time_path = edf_date_dir / f"{speaker_id}.txt"
    start_time = read_time(start_time_path)

    # the header row has been removed in func:get_rows
    rows = get_rows(input_csv)

    rows = to_list(rows)
    rows = filter_rows(rows)

    rows = append_relative_timestamp(start_time, rows)

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
