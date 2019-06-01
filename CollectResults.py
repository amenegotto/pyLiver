import pandas as pd
import glob

CSV_RESULTS_PATHS = 'c:/users/hp/Downloads/foo.csv'
CSV_OUTPUT = 'c:/users/hp/Downloads/foo55.csv'

f_out = None
first = True


def read_txt(f_path):
    global first, f_out
    files = [f for f in glob.glob(f_path + "*.txt", recursive=True)]
    for file in files:
        if 'predict' in file:
            continue

        seq = file.split('-')[2].split('.')[0]
        f = open(file, "r")
        line_header = 'path,'
        line_values = f_path + ','
        for line in f:
            if ':' in line:
                header = line.split(':')[0]
                value = line.split(':')[1].replace('\n', '').replace(',', '-')
                line_header = line_header + header + ','
                line_values = line_values + value + ','

            if 'Learning Rate' in line:
                break

        f.close()

        f_csv = open(f_path + '.csv')
        curr_line = 0
        l_first = True
        for l_csv in f_csv:

            if l_first:
                line_header = line_header + l_csv
                l_first = False

            if str(curr_line) == seq:
                line_values = line_values + l_csv

            curr_line = curr_line + 1

        if first:
            f_out.write(line_header)
            first = False

        f_out.write(line_values)


paths = pd.read_csv(CSV_RESULTS_PATHS)
for i, r in paths.iterrows():
    path = paths.at[i, 'path']
    is_fine_tuning = paths.at[i, 'fine-tuning']

    f_out = open(path + '-summary.csv', 'a')

    if not is_fine_tuning:
        read_txt(path)
    else:
        read_txt(path)

    f_out.close()