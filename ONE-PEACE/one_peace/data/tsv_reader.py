import logging
import csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVReader:
    def __init__(self, file_path, selected_cols=None, separator=","):
        self.contents = []
        with open(file_path, 'r', encoding='utf-8') as fp:
            reader = csv.reader(fp, delimiter=separator)
            headers = next(reader)
            if selected_cols is not None:
                col_ids = []
                try:
                    for v in selected_cols.split(','):
                        col_ids.append(headers.index(v))
                except ValueError:
                    logger.error(f"Column not found in headers: {v}")
                    raise
                selected_cols = col_ids
            else:
                selected_cols = list(range(len(headers)))

            for row in reader:
                if selected_cols:
                    row = [row[col_id] for col_id in selected_cols]
                self.contents.append(row)

        logger.info(f"Loaded {file_path}")
    # def __init__(self, file_path, selected_cols=None, separator=","):
    #     fp = open(file_path, encoding='utf-8')
    #     headers = fp.readline().strip().split(separator)
    #     if selected_cols is not None:
    #         col_ids = []
    #         for v in selected_cols.split(','):
    #             col_ids.append(headers.index(v))
    #         selected_cols = col_ids
    #     else:
    #         selected_cols = list(range(len(headers)))

    #     self.contents = []
    #     for row in fp:
    #         if selected_cols is not None:
    #             column_l = row.rstrip("\n").split(separator, len(headers) - 1)
    #             column_l = [column_l[col_id] for col_id in selected_cols]
    #         else:
    #             column_l = row.rstrip("\n").split(separator, len(headers) - 1)
    #         self.contents.append(column_l)

    #     logger.info("loaded {}".format(file_path))
    #     fp.close()


    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        column_l = self.contents[index]
        return column_l
