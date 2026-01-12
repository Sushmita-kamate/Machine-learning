# s={'s','u','s','h'}
# s.add('m')
# print(s)
# s.discard('u')
# print(s)
# s.remove('s')
# print(s)
# s.pop(1)
# print(s)
# s.clear()
# print(s)
# mylist=['college','university','country']
# x=frozenset(mylist)
# x[1]="college_1"
# print(mylist)



# import os
# def create_file(filename):
#     try:
#         with open(filename,'w')a f:
#             f.write('certification')
#         print("file"+filename+"successfully.")
#     except IOError:
#         print("Error"+filename)     
# import os

# def create_file(filename):
#     try:
#         with open(filename, 'w') as f:
#             f.write('Certification in Advanced Machine Learning and Introductory Deep Learning — From Foundations to Practice_Clg_Syllabus')
#         print("File " + filename + " created successfully.")
#     except IOError:
#         print("Error: could not create file " + filename)

# def read_file(filename):
#     try:
#         with open(filename, 'r') as f:
#             contents = f.read()
#             print(contents)
#     except IOError:
#         print("Error: could not read file " + filename)

# def append_file(filename, text):
#     try:
#         with open(filename, 'a') as f:
#             f.write(text)
#         print("Text appended to file " + filename + " successfully.")
#     except IOError:
#         print("Error: could not append to file " + filename)

# def rename_file(filename, new_filename):
#     try:
#         if os.path.exists(new_filename):
#             os.remove(new_filename)
#         os.rename(filename, new_filename)
#         print(f"File {filename} renamed to {new_filename} successfully.")
#     except Exception as e:
#         print(f"Error renaming file: {e}")

# def delete_file(filename):
#     try:
#         os.remove(filename)
#         print("File " + filename + " deleted successfully.")
#     except IOError:
#         print("Error: could not delete file " + filename)


# if __name__ == '__main__':
#     filename = "example.txt"
#     new_filename = "new_example.txt"

#     create_file(filename)
#     read_file(filename)
#     append_file(filename, "Python provides built-in functions to handle file operations such as reading from and writing to files. We can use the open() function to work with files.\n")
#     read_file(filename)
#     rename_file(filename, new_filename)
#  read_file(new_filename)
# delete_file(new_filename)
# import math 
# print(math.log(2,3))
# print(math.log2(16))
# print(math.log10(10000))

# import csv
# mydict = [
#     {'branch': 'COE', 'cgpa': '9.0', 'name': 'Nikhil', 'year': '2'},
#     {'branch': 'COE', 'cgpa': '9.1', 'name': 'Sanchit', 'year': '2'},
#     {'branch': 'IT', 'cgpa': '9.3', 'name': 'Aditya', 'year': '2'},
#     {'branch': 'SE', 'cgpa': '9.5', 'name': 'Sagar', 'year': '1'},
#     {'branch': 'MCE', 'cgpa': '7.8', 'name': 'Prateek', 'year': '3'},
#     {'branch': 'EP', 'cgpa': '9.1', 'name': 'Sahil', 'year': '2'} ]

# fields = ['name', 'branch', 'year', 'cgpa']

# filename = "university_records.csv"

# with open(filename, 'w', newline='') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=fields)
#     writer.writeheader()
#     writer.writerows(mydict)

# import math 
# a=math.pi/6
# print(math.sin(a))
# print(math.cos(a))
# print(math.tan(a))

# import pandas as pd
# df=pd.read_csv('university_records.csv')
# print(df.to_string())
                         
# import numpy as  np
# import time
# SIZE=100000
# L1=range(SIZE)
# L2=range(SIZE)
# A1=np.arange(SIZE)
# A2=np.arange(SIZE)
# start=time.time()
# result=[(x,y) for x,y in zip(L1,L2)]
# print((time.time()-start)*1000)
# start=time.time()
# result=A1+A2
# print((time.time()-start)*1000)



# import pandas as pd
# import numpy as np
# dict={'First score':[100,90,np.nan,95],
#        'Second score':[30,45,56,np.nan],
#        'Third  score':[np.nan,40,80,98]

# }
# df=pd.DataFrame(dict)

# import pandas as pd
# dict={'name':["sush","swat"],
#       'degree':["BCA","MBA"],
#       'score':[90,99]}
# df=pd.DataFrame(dict)
# for i,j in df.iterrows():
#     print(i,j)
#     print()