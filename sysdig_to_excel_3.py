import pandas as pd
from datetime import datetime, time

#Sample file
file1 = open('shaped-transformed-timing/CVE-2012-1823/CVE-2012-1823-1_freqvector_full.csv', 'r')
Lines = file1.readlines()
# Lines = ['CVE-2012-1823-1_freqvector_full.csv']
trace_number = 4
cveId = 'CVE-2022-21449'
directory = 'shaped-transformed-timing/' + cveId + '/' + cveId + '-' + str(trace_number)
directory_2 = 'shaped-transformed-timing/' + cveId + '/' + 'cve_2022-21449_' + str(trace_number)
#file2 = open(directory_2 + '_syscalls.txt', 'r')

#Conversion file
file2 = open('spring_4shell_f1.txt', 'r')
Lines2 = file2.readlines()
list_of_rows = []
list_of_cves = []


def get_sec(time_str):
    """Get seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


one_minute = 600
three_minute = 1800
four_minute = 2400
# train_autoencoder_path = 'C:/Users/12103/PycharmProjects/cdl_remade/venv/train_autoencoder/'
# train_autoencoder_path = 'C:/Users/sadun/PycharmProjects/cdl_remade/venv/train_autoencoder/'
# df = pd.DataFrame()
df2 = pd.DataFrame()
# list_of_values = [1,2,3,4]
list_of_values = [1]
normal = 1
anomaly = 0
list_of_all = []
list_of_system_calls = []
list_of_stripped_system_call = []
list_of_rows = []
# list_of_rows = []
list_of_values = [0] * 556

cve_four_counter = 1
line_number = 1
for line in Lines:
    content = line.strip()
    # print(cve)

    if line_number == 1:
        # print(content)
        list_of_system_calls = content.split(',')[1:]
        list_of_system_calls.insert(0, 'timestamp')
        # print(list_of_system_calls)
        list_of_rows.append(list_of_system_calls)
        line_number += 1
        break

for element in list_of_system_calls:
    stripped_system_call = element.split('[')[0]
    # print(element)
    # print(stripped_system_call)
    list_of_stripped_system_call.append(stripped_system_call)

# print("FF")
# print(list_of_system_calls[1], list_of_stripped_system_call[1])

appended_microsecond = 0
one_tenth_of_a_second = 0.10

first_record = True
zero_minute = 0
zero_second = 0
timestamp_lower_bound_int = 0
timestamp_upper_bound_int = 0

for line2 in Lines2:
    content2 = line2.strip()
    content2_split = content2.split(' ')
    print(content2)
    split_time = content2_split[0]
    system_call = content2_split[1]
    print(split_time)
    print(system_call)
    split_time_componenets = split_time.split(':')
    hour = int(split_time_componenets[0])
    minute = int(split_time_componenets[1])
    # minute = minute +7
    second_float = split_time_componenets[2]
    split_time_second = second_float.split(".")
    second_integer = int(split_time_second[0])
    microsecond = int(split_time_second[1][:6])
    timestamp_in_string = str(split_time_componenets[0]) + str(split_time_componenets[1]) + str(
        split_time_second[0]) + str(split_time_second[1][:2])
    print("TIMESTAMP:", timestamp_in_string)
    print(microsecond)
    # new_microsend = microsecond + 100000
    my_time = time(hour, minute, second_integer, microsecond)
    # my_time = datetime("%H:%M:%S.%f")
    print(my_time)
    my_time_integer = int(my_time.strftime("%H%M%S"))
    print(my_time_integer)
    timestamp_in_integer = float(timestamp_in_string)
    first_timestamp = 0

    if first_record is True:
        # timestamp_in_integer = int(timestamp_in_string)
        first_record = False
        timestamp_lower_bound_int = float(timestamp_in_integer)
        timestamp_upper_bound_int = float(timestamp_lower_bound_int + 10)
        # list_of_values =

    if first_record is False:
        first_time = True
        # print("OUT OF LOOP INCREMENT OLD")
        # print(timestamp_in_integer)
        # print(timestamp_lower_bound_int)
        # print(timestamp_upper_bound_int)
        if timestamp_in_integer > timestamp_upper_bound_int:
            print("DIFF:", timestamp_in_integer - timestamp_upper_bound_int)
            # print("IN INCREMENT OLD")
            # print(timestamp_in_integer)
            # print(timestamp_lower_bound_int)
            # print(timestamp_upper_bound_int)

            while (timestamp_upper_bound_int < timestamp_in_integer):
                # list_of_values = [0]*556

                # if first_time is True:
                #     list_of_values[0]=timestamp_lower_bound_int
                #     first_time = False
                # else:
                #     list_of_values[0]=timestamp_upper_bound_int

                list_of_values[0] = timestamp_lower_bound_int
                list_of_rows.append(list_of_values)
                list_of_values = [0] * 556
                timestamp_lower_bound_int = timestamp_upper_bound_int
                if second_integer == 0:
                    print("second_integer:", second_integer)
                    timestamp_upper_bound_int = timestamp_in_integer + 10

                else:
                    timestamp_upper_bound_int += 10

            # print("IN INCREMENT NEW")
            # print(timestamp_in_integer)
            # print(timestamp_lower_bound_int)
            # print(timestamp_upper_bound_int)

        elif timestamp_in_integer < timestamp_upper_bound_int and timestamp_in_integer >= timestamp_lower_bound_int:
            list_of_values[0] = timestamp_lower_bound_int
            try:
                index_of_system_call = list_of_stripped_system_call.index(system_call)
                list_of_values[index_of_system_call] += 1
            except ValueError:
                pass

df2 = pd.DataFrame(list_of_rows)
print(df2)
#df2.to_csv(directory + '_freqvector_full.csv', sep=',', encoding='utf-8', index=False, header=False)
df2.to_csv('spring_4shell_f1.csv', sep=',', encoding='utf-8', index=False, header=False)