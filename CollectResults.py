import pandas as pd
import glob

CSV_RESULTS_PATHS = '/mnt/data/results/index.csv'

f_out = None

def read_txt(f_path, is_fine_tuning):
    global first, f_out
    first = True
    files = [f for f in glob.glob(f_path + "*.txt", recursive=True)]
    for file in files:
        if 'predict' in file:
            continue

        if is_fine_tuning:
            seq = file.split('-')[3].split('.')[0]
        else:
            seq = file.split('-')[2].split('.')[0]

        f = open(file, "r")
        line_header = 'path,'
        line_values = f_path + ','
        ignore = False
        for line in f:

            if 'Kappa Score' in line:
                line_header = line_header + 'Kappa Score,'
                line_values = line_values + line.split('=')[1].replace('\n','').strip() + ','
                break

            if ignore:
                continue
            
            if ':' in line:
                header = line.split(':')[0]
                value = line.split(':')[1].replace('\n', '').replace(',', '-')
                line_header = line_header + header + ','
                line_values = line_values + value + ','

            if 'Learning Rate' in line:
                ignore = True

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
            if line_header[-1] != '\n':
                line_header = line_header + '\n'
            f_out.write(line_header)
            first = False

        if line_values[-1] != '\n':
            line_values = line_values + '\n'

        f_out.write(line_values)


paths = pd.read_csv(CSV_RESULTS_PATHS)
for i, r in paths.iterrows():
    path = paths.at[i, 'path']
    is_fine_tuning = paths.at[i, 'fine-tuning']

    f_out = open(path + '-summary.csv', 'a')

    read_txt(path, is_fine_tuning)

    f_out.close()
