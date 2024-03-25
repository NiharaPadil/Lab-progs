-- Consider the schema for Airline Database: 
-- Flights (Flight_num, Source, Destination, Distance, Departs, Arrives, Price) 
-- Aircraft (Aid, Aname, Cruising_range) 
-- Certified (Emp_id, Aid) 
-- Employees (Emp_id, Ename, Salary) 
-- Note: The Employees relation describes pilots and other kinds of employees as well; Every pilot is certified for some aircraft, and only pilots are certified to fly. 


-- Write SQL queries to 
-- 1.   Find the names of aircraft such that all pilots certified to operate them have salaries more than Rs.80, 000. 
-- 2. For each pilot who is certified for more than three aircrafts, find the emp_id and the maximum cruisingrange of the aircraft for which she or he is certified. 
-- 3. Find the names of pilots whose salary is less than the price of the cheapest route from Bengaluru to Mumbai. 
-- 4.   Find the aids of all aircraft that can be used on routes from Bengaluru to New Delhi. 
-- 5.   Find the employee name and salary earning second highest salary. 


create schema airline;
use airline;


CREATE TABLE FLIGHT(
     	FLIGHT_NUM INT,
     	SOURCE VARCHAR(20),
     	DESTINATION VARCHAR(20),
    	DISTANCE INT,
    	DEPARTS VARCHAR(10),
    	ARRIVES VARCHAR(10),
     	PRICE INT,
    	PRIMARY KEY (FLIGHT_NUM) );


CREATE TABLE EMPLOYEES(
     		EID INT,
    		ENAME VARCHAR(20),
    		SALARY INT,
    		PRIMARY KEY (EID) );
            
CREATE TABLE AIRCRAFT(
    	AID INT,
    	ANAME VARCHAR(20),
    	CRUISINGRANGE INT,
    	PRIMARY KEY (AID) );

CREATE TABLE CERTIFIED(
     		EID INT,
    		 AID INT,
    		 PRIMARY KEY (EID,AID),
     		FOREIGN KEY (EID) REFERENCES EMPLOYEES (EID),
     		FOREIGN KEY (AID) REFERENCES AIRCRAFT (AID) );


INSERT INTO FLIGHT VALUES(1,'BANGALORE','MANGALORE',300,10.45,12.00,10000);
INSERT INTO FLIGHT VALUES(2,'BANGALORE','DELHI',5000,12.15,4.30,25000);
INSERT INTO FLIGHT VALUES(3,'BANGALORE','MUMBAI',3500,2.15,5.25,30000);
INSERT INTO FLIGHT VALUES(4,'DELHI','MUMBAI',4500,10.15,12.05,35000);
INSERT INTO FLIGHT VALUES(5,'DELHI','FRANKFURT',18000,7.15,5.30,90000);
INSERT INTO FLIGHT VALUES(6,'BANGALORE','FRANKFURT',19500,10.00,7.45,95000);
INSERT INTO FLIGHT VALUES(7,'BANGALORE','FRANKFURT',17000,12.00,6.30,99000);


INSERT INTO AIRCRAFT VALUES(123,'AIRBUS',1000);
INSERT INTO AIRCRAFT VALUES(302,'BOEING',5000);
INSERT INTO AIRCRAFT VALUES(306,'JET01',5000);
INSERT INTO AIRCRAFT VALUES(378,'AIRBUS380',8000);
INSERT INTO AIRCRAFT VALUES(456,'AIRCRAFT',500);
INSERT INTO AIRCRAFT VALUES(789,'AIRCRAFT02',800);
INSERT INTO AIRCRAFT VALUES(951,'AIRCRAFT03',1000);



INSERT INTO EMPLOYEES VALUES(1,'AJAY',30000);
INSERT INTO EMPLOYEES VALUES(2,'AJITH',85000);
INSERT INTO EMPLOYEES VALUES(3,'ARNAB',50000);
INSERT INTO EMPLOYEES VALUES(4,'HARRY',45000);
INSERT INTO EMPLOYEES VALUES(5,'ARUN',90000);
INSERT INTO EMPLOYEES VALUES(6,'JOSH',75000);
INSERT INTO EMPLOYEES VALUES(7,'RAM',100000);

INSERT INTO CERTIFIED VALUES(1,123);
INSERT INTO CERTIFIED VALUES(2,123);
INSERT INTO CERTIFIED VALUES(1,302);
INSERT INTO CERTIFIED VALUES(5,302);
INSERT INTO CERTIFIED VALUES(7,302);
INSERT INTO CERTIFIED VALUES(1,306);
INSERT INTO CERTIFIED VALUES(2,306);
INSERT INTO CERTIFIED VALUES(1,378);
INSERT INTO CERTIFIED VALUES(2,378);
INSERT INTO CERTIFIED VALUES(4,378);
INSERT INTO CERTIFIED VALUES(3,456);
INSERT INTO CERTIFIED VALUES(6,456);
INSERT INTO CERTIFIED VALUES(1,789);
INSERT INTO CERTIFIED VALUES(5,789);
INSERT INTO CERTIFIED VALUES(6,789);
INSERT INTO CERTIFIED VALUES(1,951);
INSERT INTO CERTIFIED VALUES(3,951);



-- QUERY 1
select a.aid,a.aname
from employees e, certified c , aircraft a
where a.aid=c.aid and c.eid=e.eid and e.salary>80000;



-- QUERY 2

 SELECT c.eid,MAX(cruisingrange)
FROM certified c,aircraft a
WHERE c.aid=a.aid
GROUP BY c.eid
HAVING COUNT(*)>3;


-- QUERY 3

 SELECT DISTINCT e.ename
      FROM employees e
      WHERE e.salary<=
      (SELECT MIN(f.price)
      FROM flight f
      WHERE f.Source='BANGALORE'
       AND f.Destination='MUMBAI');

-- QUERY 4

SELECT a.aid
     FROM aircraft a
     WHERE a.cruisingrange>
     (SELECT MIN(f.distance)
     FROM flight f
     WHERE f.Source='BANGALORE'
     AND f.Destination='DELHI');

-- QUERY 5

SELECT ename,salary from employees where salary=(select MAX(salary) 
FROM employees 
WHERE salary <(select max(salary)from employees));









