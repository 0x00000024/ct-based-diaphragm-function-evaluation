import pandas as pd
import settings
from colorama import Fore

in_csv_url = 'https://ct.eny.li/10017162/in/211109-113136/points.csv'
ex_csv_url = 'https://ct.eny.li/10017162/ex/211109-113230/points.csv'

df_in = pd.read_csv(in_csv_url)
df_ex = pd.read_csv(ex_csv_url)

start_image_num = df_in.iloc[0]['image_number']
end_image_num = df_in.iloc[-1]['image_number']

dist_list = []
total_pixel_num = 0
while start_image_num <= end_image_num:
    curr_image_df_in = df_in.loc[df_in['image_number'] == start_image_num]
    curr_image_df_ex = df_ex.loc[df_ex['image_number'] == start_image_num]
    start_x_value = max(curr_image_df_in['x_value'].iloc[0],
                        curr_image_df_ex['x_value'].iloc[0])
    stop_x_value = min(curr_image_df_in['x_value'].iloc[-1],
                       curr_image_df_ex['x_value'].iloc[-1])

    curr_in_dict = {}
    curr_ex_dict = {}
    for index, row in curr_image_df_in.iterrows():
        curr_in_dict[row['x_value']] = row['y_value']
    for index, row in curr_image_df_ex.iterrows():
        curr_ex_dict[row['x_value']] = row['y_value']

    curr_pixel_num = 0
    for x in range(start_x_value, stop_x_value, 1):
        if x in curr_in_dict and x in curr_ex_dict:
            if curr_ex_dict[x] < curr_in_dict[x]:
                print(
                    Fore.RED +
                    '[Error] The position of the diaphragm during exhalation is below the position of the diaphragm'
                    ' during inhalation')
            else:
                dist = curr_ex_dict[x] - curr_in_dict[x]
                dist_list.append(dist)
                curr_pixel_num += dist
        else:
            print(Fore.RED + '[Linear interpolation error] Uncovered frame:',
                  x)

    print(
        Fore.GREEN +
        f'The number of pixels between the diaphragm at different positions of the #{start_image_num} slice: '
        + str(curr_pixel_num))

    total_pixel_num += curr_pixel_num

    start_image_num += 1

print(
    Fore.GREEN +
    'The volume (cm^3) between the diaphragms in different positions:',
    total_pixel_num * settings.row_spacing * settings.col_spacing *
    settings.thickness)

print(
    Fore.GREEN +
    'The longest distance (cm) between the diaphragms in different positions:',
    max(dist_list) * settings.col_spacing)
print(
    Fore.GREEN +
    'The shortest distance (cm) between the diaphragms in different positions:',
    min(dist_list) * settings.col_spacing)
