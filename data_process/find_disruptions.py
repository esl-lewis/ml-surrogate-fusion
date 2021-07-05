from numpy.lib.function_base import place
import pandas as pd
import numpy as np
import re

def extract_disrupt_time(pulse_row):
    pre_comment = pulse_row["prepulse_comment"]
    post_comment = pulse_row["postpulse_comment"]
    num = pulse_row["pulse_number"]

    comment = pre_comment + post_comment

    placehold = []

    #disrupt_time = re.findall(r"[0-9]{2,6}", comment)
    #if disrupt_time:
    #    placehold = disrupt_time

    disrupt_time = re.findall(r"[0-9]{2,6}s", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"[0-9]{2,6} s", comment)
    if disrupt_time:
        placehold = disrupt_time

    #disrupt_time = re.findall(r"[0-9]+[.][0-9]+", comment)
    #if disrupt_time:
    #    placehold = disrupt_time

    disrupt_time = re.findall(r"[0-9]+[.][0-9]+s", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"[0-9]+[.][0-9]+ s", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"[0-9]+[.]s", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"[0-9]+[.][0-9]+ sec", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"at [0-9]+[.][0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"[0-9]+s disrupt", comment)
    if disrupt_time:
        placehold = disrupt_time
    
    disrupt_time = re.findall(r"lock at [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"ock @ [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"rupted [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"rupt~[0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"rupts~[0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"isrupt at [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"isrupts later at [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"upt @ [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"upts @ [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"upts @[0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"upted @[0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"isrupts ~[0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"isrupt. [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"isrupts at [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"isrupted at [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"isrupted @ [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"isruption~[0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"isrupted ~ [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"ruption [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"ruption @ [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"ruption @[0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    disrupt_time = re.findall(r"ruption at [0-9]+", comment)
    if disrupt_time:
        placehold = disrupt_time

    # placehold = [value.strip("secat@upt").strip() for value in placehold]
    placehold = [re.sub("[^0-9.]+", "", value) for value in placehold]

    disrupt_time_list = placehold

    if len(disrupt_time_list) > 1:
        print("Multiple time values ambiguous for pulse", num)
        print(disrupt_time_list)
    if len(disrupt_time_list) == 1:
        pulse_row["disruption time"] = disrupt_time_list[0]
    # else:
    #    print("No time value found for pulse", num)
    return pulse_row

def filter_against_no_disrupt(pulse_row):
    # pulse_row is a df or series row
    disrupt = True

    pre_comment = pulse_row["prepulse_comment"]
    post_comment = pulse_row["postpulse_comment"]

    negations = ["Didn't disrupt", "didn't disrupt", "no disruption", "not disrupt"]

    if any(negate_phrase in pre_comment for negate_phrase in negations):
        disrupt = False
    elif any(negate_phrase in post_comment for negate_phrase in negations):
        disrupt = False

    pulse_row["Disrupted?"] = disrupt
    # print(pulse_row)
    return pulse_row

def filter_for_disrupt(pulse_row):
    disrupt =False
    pre_comment = pulse_row["prepulse_comment"]
    post_comment = pulse_row["postpulse_comment"]

    disruption_present = ["disrupt", "Disrupt","disruption","Disruption","disrupted","Disrupted"]
    if any(negate_phrase in pre_comment for negate_phrase in disruption_present):
        disrupt = True
    elif any(negate_phrase in post_comment for negate_phrase in disruption_present):
        disrupt = True
    
    pulse_row["Disrupted?"] = disrupt
    return pulse_row

# =================================================================================================
# filter for disruptions 
# NB 1922 pulses found initially by searching from 1/6/2010 onwards
test_file = "pulse_data.csv"

df_pulses = pd.read_csv(test_file, engine="python")
df_pulses = df_pulses.fillna("emptyfill")

df_pulses = df_pulses.apply(filter_for_disrupt, axis=1)
only_disruptions = df_pulses.drop(df_pulses[df_pulses["Disrupted?"] == False].index)
print(df_pulses.shape[0])
print(only_disruptions.shape[0])

# =================================================================================================
# filtering out not disruptions 


only_disruptions = only_disruptions.apply(filter_against_no_disrupt, axis=1)
only_disruptions = only_disruptions.drop(only_disruptions[only_disruptions["Disrupted?"] == False].index)

print(only_disruptions.shape[0])

# TODO need to slightly alter filters so also capturing after decimal point

disruption_numbers = only_disruptions["Disrupted?"].value_counts().reset_index()
print(disruption_numbers)

"""
only_disruptions.to_csv("disruption_pulses.csv", index=False)



# =================================================================================================
# extracting disruption time and type

only_disruptions = pd.read_csv("disruption_pulses.csv")
only_disruptions = only_disruptions.drop(["Disrupted?"], axis=1)
only_disruptions = only_disruptions.apply(extract_disrupt_time, axis=1)
# print(only_disruptions.head(35))
print(
    "we have disruption value times for:",
    only_disruptions.count(axis=0)["disruption time"],
    "pulses",
)

only_disruptions = only_disruptions.dropna()

print(
    "we have disruption value times for:",
    only_disruptions.count(axis=0)["disruption time"],
    "pulses",
)


print(only_disruptions.head(20))
only_disruptions.to_csv("disruption_times.csv", index=False)
"""
# possible to extract causes eg locked mode, but not worth it
