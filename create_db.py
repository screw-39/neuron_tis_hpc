import sqlite3

#connect to database
conn = sqlite3.connect('./DB/WAYCHIA_SYMMETRY.db')

#creat new table
conn.execute('''CREATE TABLE TEST_PARAMETER(        
        TEST_ID         INT      NOT NULL,
        ELECTRODE_NUM   INT      NOT NULL,
        THETA           REAL     NOT NULL,
        RO              REAL     NOT NULL,
        ROLL            REAL     NOT NULL,
        AMPLITUDE       REAL     NOT NULL,
        FREQUENCY       REAL     NOT NULL,
        DELTA           REAL     NOT NULL,
        PRIMARY KEY (TEST_ID)
        );''')
#submit change
conn.commit()

conn.execute('''CREATE TABLE TEST_VOLTAGE(      
        TEST_ID INT     NOT NULL,
        TIME    REAL    NOT NULL,
        VOLTAGE REAL    NOT NULL,
        PRIMARY KEY (TEST_ID, TIME),
        FOREIGN KEY (TEST_ID) REFERENCES TEST_PARAMETER(TEST_ID)
        );''')
conn.commit()

conn.execute('''CREATE TABLE ELECTRODE_PARAMETER          
        (TEST_ID        INT      NOT NULL,
        ELECTRODE_ID    INT      NOT NULL,
        X               REAL     NOT NULL,
        Y               REAL     NOT NULL,
        Z               REAL     NOT NULL,
        PRIMARY KEY (TEST_ID, ELECTRODE_ID),
        FOREIGN KEY (TEST_ID) REFERENCES TEST_PARAMETER(TEST_ID)     
        );''')
conn.commit()
conn.close()