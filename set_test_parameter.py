import sqlite3

#connect to database
conn = sqlite3.connect('./DB/3D_SYMMETRY.db')
c = conn.cursor()

#set current parameter
frequency = 1000
delta = 20
amp = 0.276*1e5
#2 electrode : , 4 electrode : 0.276*1e5, 6 electrode : 

#set electrode parameter
TEST_ID = 0
for j in range(0, 360, 10):
        for k in range(0, 360, 10):
            for l in range(0, 360, 10):
                c.execute(f'''INSERT INTO TEST_PARAMETER (TEST_ID,THETA,RO,ROLL,AMPLITUDE,FREQUENCY,DELTA)
                    VALUES ({TEST_ID}, {k}, {l}, {j}, {amp}, {frequency}, {delta} );''')
                
#submit change
conn.commit()
conn.close()