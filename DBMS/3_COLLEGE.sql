-- Consider the schema for College Database: 
-- STUDENT(USN, SName, Address, Phone, Gender) 
-- SEMSEC(SSID, Sem, Sec) 
-- CLASS(USN, SSID) 
-- COURSE(Subcode, Title, Sem, Credits) 
-- IAMARKS(USN, Subcode, SSID, Test1, Test2, Test3, FinalIA) 

-- Write SQL queries to 
-- 1. List all the student details studying in fourth semester „C‟ section. 
-- 2. Compute the total number of male and female students in each semester and in each section. 
-- 3. Create a view of Test1 marks of student USN “4SF20CD001” in all Courses. 
-- 4. Calculate the FinalIA (average of three test marks) and update the corresponding table for  all students. 
-- 5. Categorize students based on the following criterion: 
	-- If FinalIA = 45 to 50 then CAT = “Outstanding” 
	-- If FinalIA= 40 to 45 then CAT= “Good” 
	-- If FinalIA = 30 to 40 then CAT = “Average” 
	-- If FinalIA< 30 then CAT = “Weak” 
-- Give these details only for 8th semester A, B, and C section students.

create schema college;
use college;

-- Create table STUDENT
CREATE TABLE STUDENT (
    USN VARCHAR(25) PRIMARY KEY,
    SName VARCHAR(25),
    Address VARCHAR(25),
    Phone VARCHAR(25),
    Gender VARCHAR(10)
);

-- Create table SEMSEC
CREATE TABLE SEMSEC (
    SSID VARCHAR(25) PRIMARY KEY,
    Sem VARCHAR(25),
    Sec VARCHAR(25)
);

-- Create table CLASS
CREATE TABLE CLASS (
    USN VARCHAR(25),
    SSID VARCHAR(25),
    PRIMARY KEY (USN, SSID),
    FOREIGN KEY (USN) REFERENCES STUDENT(USN) ON DELETE CASCADE,
    FOREIGN KEY (SSID) REFERENCES SEMSEC(SSID) ON DELETE CASCADE
);

-- Create table COURSE
CREATE TABLE COURSE (
    Subcode VARCHAR(25) PRIMARY KEY,
    Title VARCHAR(25),
    Sem VARCHAR(25),
    Credits INT
);

-- Create table IAMARKS
CREATE TABLE IAMARKS (
    USN VARCHAR(25),
    Subcode VARCHAR(25),
    SSID VARCHAR(25),
    Test1 INT,
    Test2 INT,
    Test3 INT,
    FinalIA INT,
    PRIMARY KEY (USN, Subcode, SSID),
    FOREIGN KEY (USN) REFERENCES STUDENT(USN) ON DELETE CASCADE,
    FOREIGN KEY (Subcode) REFERENCES COURSE(Subcode) ON DELETE CASCADE,
    FOREIGN KEY (SSID) REFERENCES SEMSEC(SSID) ON DELETE CASCADE
);


-- Insert values into STUDENT table
INSERT INTO STUDENT VALUES ('S001', 'Alice', 'Address 1', '1234567890', 'Female');
INSERT INTO STUDENT VALUES('S002', 'Bob', 'Address 2', '2345678901', 'Male');
INSERT INTO STUDENT VALUES('S003', 'Charlie', 'Address 3', '3456789012', 'Male');
INSERT INTO STUDENT VALUES('S004', 'Diana', 'Address 4', '4567890123', 'Female');
INSERT INTO STUDENT VALUES('S005', 'Evan', 'Address 5', '5678901234', 'Male');

-- Insert values into SEMSEC table
INSERT INTO SEMSEC VALUES ('SS001', '4', 'A');
INSERT INTO SEMSEC VALUES ('SS002', '4', 'B');
INSERT INTO SEMSEC VALUES ('SS003', '4', 'C');
INSERT INTO SEMSEC VALUES ('SS004', '8', 'A');
INSERT INTO SEMSEC VALUES ('SS005', '8', 'B');

-- Insert values into CLASS table
INSERT INTO CLASS VALUES ('S001', 'SS001');
INSERT INTO CLASS VALUES ('S002', 'SS002');
INSERT INTO CLASS VALUES ('S003', 'SS003');
INSERT INTO CLASS VALUES ('S004', 'SS004');
INSERT INTO CLASS VALUES ('S005', 'SS005');

-- Insert values into COURSE table
INSERT INTO COURSE VALUES ('C001', 'Course 1', '4', 3);
INSERT INTO COURSE VALUES ('C002', 'Course 2', '4', 4);
INSERT INTO COURSE VALUES ('C003', 'Course 3', '8', 3);
INSERT INTO COURSE VALUES('C004', 'Course 4', '8', 4);
INSERT INTO COURSE VALUES('C005', 'Course 5', '8', 3);

-- Insert values into IAMARKS table
INSERT INTO IAMARKS VALUES ('S001', 'C001', 'SS001', 80, 85, 90, 85);
INSERT INTO IAMARKS VALUES ('S002', 'C002', 'SS002', 75, 80, 85, 80);
INSERT INTO IAMARKS VALUES ('S003', 'C003', 'SS003', 70, 75, 80, 75);
INSERT INTO IAMARKS VALUES ('S004', 'C004', 'SS004', 65, 70, 75, 70);
INSERT INTO IAMARKS VALUES ('S005', 'C005', 'SS005', 60, 65, 70, 65);


-- QUERY1
SELECT S.* 
FROM STUDENT S
JOIN CLASS C ON C.USN=S.USN
JOIN SEMSEC SE ON SE.SSID =C.SSID 
WHERE SEM ='4' AND SEC ='C'; 

-- QUERY 2 
SELECT SE.SEM , SE.SEC,S.GENDER,COUNT(*)AS TOTAL
FROM STUDENT S
JOIN CLASS C ON C.USN=S.USN
JOIN SEMSEC SE ON SE.SSID =C.SSID 
GROUP BY SE.SEM , SE.SEC,S.GENDER order by SE.SEM;

-- QUERY 3 

CREATE VIEW TEST_RES AS 
 SELECT I.SUBCODE,I.TEST1
 FROM IAMARKS I 
 WHERE I.USN ='S002';
 
 SELECT * FROM TEST_RES;
			
-- QUERY 4

UPDATE IAMARKS
SET FINALIA=GREATEST((TEST1+TEST2),(TEST2+TEST3),(TEST1+TEST3))/2;

SELECT * FROM IAMARKS;

-- QUERY 5 

SELECT S.* , CASE
	   WHEN I.FinalIA BETWEEN 90 AND 100 THEN 'Outstanding'
           WHEN I.FinalIA BETWEEN 80 AND 490 THEN 'Good'
           WHEN I.FinalIA BETWEEN 70 AND 80 THEN 'Average'
           WHEN I.FinalIA < 30 THEN 'Weak'
           ELSE 'Undefined'
           END AS CAT
FROM STUDENT S, SEMSEC SS,iamarks I 
WHERE S.USN=I.USN AND I.SSID=SS.SSID 
AND SS.SEM='8' AND SS.SEC IN ('A','B','C');

            
            
            
