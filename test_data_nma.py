import pandas as pd
import numpy as np
import uuid

days = 16
selected_cols = ['TimeSlot: {}'.format(j/2.0) for j in range(48)]

#{17, 17, 68, 68, 68, 68, 26, 26, 45, 45, 19, 19, 69, 69, 69, 69, 20, 20, 20, 20, 11, 11, 11, 11, 0, 0, 0, 0, 104, 104, 276, 276, 318, 318, 269, 269, 122, 122, 0, 0, 0, 0, 7, 7, 7, 7, 17, 17}
dists_student = {"1A": int(28/4.0), "1B": int(38/4.0), "1C": int(41/4.0), "2A": int(76/4.0), "2B": int(199/4.0), "2C": int(488/4.0), "3A": int(416/4.0), "3B": int(169/4.0), "3C": int(101/4.0)}
total_student = dists_student.values()
total_student = list(total_student)
total_student = sum(total_student)


#distribution_timezone = matrix_array
distribution_timezone = pd.read_csv('ProjectTimesLondon.csv',header = None)
distribution_timezone = np.array(distribution_timezone.values)
time_zones_names = ["1A", "1B", "1C", "2A", "2B", "2C", "3A", "3B", "3C"]
combine_timezone = {}
for i in range(distribution_timezone.shape[0]):
    combine_timezone[time_zones_names[i]] = distribution_timezone[i,:]


#matrix_student = np.zeros((total_student,48))
cum = 0
matrix_student = []
for key in dists_student.keys():
    partial_students = dists_student[key]
    time_availability = combine_timezone[key]
    for i in range(partial_students):
        row = {'Group_name':cum, 'Type': 'Student', 'Timezone': key}
        hash = uuid.uuid4()
        row['hash'] = hash.hex
        row['Availability'] = time_availability
        cum = cum + 1
        matrix_student.append(row)
    #matrix_student[cum:cum+partial_students,:] = np.stack([time_availability for i in range(partial_students)], axis=0)
    #cum = cum + partial_students

xls = pd.ExcelFile('Mentor List - All.xlsx')
df1 = pd.read_excel(xls, 'Confirmed Mentors - Full Applic')
df_mentors = df1[['Last name *', 'First name *', 'Email address? *','How many hours total (across three weeks) would you be able to commit? *']]

total_mentors =  278 #len(df_mentors)
dists_mentor = [0, 3, 30,47   ,86   ,55   ,156,  46 ,0]
probability_mentor = np.array(dists_mentor)/sum(np.array(dists_mentor))
#matrix_mentors = []
cum = 0
for i in range(total_mentors):
    group_number = np.random.choice([1,2,3],1, p=(0.4,0.4,0.2))
    timezone_mentor = np.random.choice(time_zones_names,1, p=probability_mentor)
    time_availability = combine_timezone[timezone_mentor[0]]
    probability_time = np.array(time_availability)/sum(np.array(time_availability))
    choice_one_time = np.random.choice([m for m in range(len(time_availability))],group_number,p=probability_time)
    for j in range(group_number[0]):
        hash = uuid.uuid4()
        matrix_single_time_slot = np.zeros(len(time_availability))
        matrix_single_time_slot[choice_one_time[j]] = 1.0
        row = {'Group_name':cum, 'Type': 'Mentor', 'Timezone': timezone_mentor[0]}
        row['hash'] = hash.hex
        row['Availability'] = matrix_single_time_slot
        matrix_student.append(row)
    cum = cum + 1

#matrix_mentors = np.stack(matrix_mentors, axis=0)
#combine = [matrix_student, matrix_mentors]
#combine = np.concatenate(combine, axis=0)
df = pd.DataFrame(matrix_student) #, columns=list(selected_cols))
print(len(df), df.keys())
df.to_pickle('fake_data_nma.csv')
