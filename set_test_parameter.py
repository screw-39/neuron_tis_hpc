import sqlite3

#connect to database
conn = sqlite3.connect('./DB/WAYCHIA_SYMMETRY.db')
c = conn.cursor()

#set current parameter
frequency = 1000
delta = 20
amps = [378.05*1e4, 189.03*1e4, 41.44*1e4]
#2 electrode : , 4 electrode : 0.276*1e5, 6 electrode : 0.1815*1e5

#waychia:
    #spike happened with 6 electrodes (R = 100, D = 20)(nA / 0.001 uA): 41.44*1e4
    #spike happened with 4 electrodes (R = 100, D = 20)(nA / 0.001 uA): 189.03*1e4
    #spike happened with 2 electrodes (R = 100, D = 20)(nA / 0.001 uA): 378.05*1e4

#set electrode parameter
TEST_ID = 0
for num in range(3):
    amp = amps[num]
    num = 2*num + 2
    for j in range(0, 360, 10):
            for k in range(0, 360, 10):
                for l in range(0, 360, 10):
                    c.execute(f'''INSERT INTO TEST_PARAMETER (TEST_ID,ELECTRODE_NUM,THETA,RO,ROLL,AMPLITUDE,FREQUENCY,DELTA)
                        VALUES ({TEST_ID}, {num}, {k}, {l}, {j}, {amp}, {frequency}, {delta} );''')
                    TEST_ID += 1
                
#submit change
conn.commit()
conn.close()