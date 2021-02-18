import psycopg2

mydb = psycopg2.connect(host="localhost",user="root",password="parikshit14")
# print(mydb)
if mydb:
    print("connection succesfull")
else:
    print("unsuccesful")
