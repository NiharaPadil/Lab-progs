-- Consider the schema for Company Database: 
-- EMPLOYEE (Eid, Name, Address, Gender, Salary, SuperEid, Dno) 
-- DEPARTMENT (Dnum, Dname, DMgr_id, Mgr_start_date) 
-- DLOCATION (Dno, Dlocation) 
-- PROJECT (Pnum, Pname, Plocation, Dno) 
-- WORKS_ON (Eid, Pno, Hours) 
-- DEPENDENT (Empid, Dep_name, Gender, Bdate, Relationship) 


-- Write SQL queries to 
-- 1.   Make a list of all project numbers for projects that involve an employee whose name is “Rahul”, either as a worker or as a manager of the department that controls the project. 
-- 2.   Show the resulting salaries if every employee working on the “IoT” project is given a 10 percent raise. 
-- 3.   Find the  sum of the  salaries of all employees  of the  “Accounts” department,  as well as the maximum salary, the minimum salary, and the average salary in this department. 
-- 4. Retrieve the name of each employee who works on all the projects controlled by department number 5 (use NOT EXISTS operator). 
-- 5. Create a view Dept_info that gives details of department name, Number of employees and total salary of each department. 

create schema company;
use company;

CREATE TABLE employee(
Eid INT PRIMARY KEY,
name VARCHAR(20),
address VARCHAR(20),
Gender CHAR(1) CHECK(Gender ='M' OR Gender ='F'),
salary int,
SuperEid int,
foreign key (supereid) references EMPLOYEE(Eid),
dno int );

INSERT INTO DEPENDENT VALUES (1, 'Dependent1', 'M', '1990-01-01', 'Child');
INSERT INTO DEPENDENT VALUES (2, 'Dependent2', 'F', '1992-03-15', 'Spouse');
INSERT INTO DEPENDENT VALUES (3, 'Dependent3', 'F', '1988-08-25', 'Child');
INSERT INTO DEPENDENT VALUES (4, 'Dependent4', 'M', '1995-05-10', 'Child');
INSERT INTO DEPENDENT VALUES (5, 'Dependent5', 'M', '1997-12-20', 'Child');


INSERT INTO employee VALUES(1, 'Rahul' , 'Mangaluru', 'M',35000,1,NULL);
INSERT INTO employee VALUES(2, 'Sahana', 'Mangaluru', 'F',35000,1,NULL);
INSERT INTO employee VALUES(3, 'Sagar', 'Bengaluru', 'M',35000,1,NULL);
INSERT INTO employee VALUES(4, 'Sagarik', 'Mangaluru', 'M',35000,1,NULL);
INSERT INTO employee VALUES(5, 'Sajaan', 'Mysore', 'M',600000,1,NULL);


CREATE TABLE department (
Dnum int PRIMARY KEY,
dname VARCHAR(10) ,
Dmgr_id int REFERENCES employee(Eid),
Mgr_start_date date);

-- '2-Nov-2007'
INSERT INTO department VALUES(1,'CSE',1,'2007-11-02');
INSERT INTO department VALUES(2,'IOT',2,'2007-11-02');
INSERT INTO department VALUES(3,'Account',2,'2017-11-02');
INSERT INTO department VALUES(4,'ISE',1,'2000-11-02');
INSERT INTO department VALUES(5,'Finance',1,'2001-11-03');



ALTER TABLE employee ADD CONSTRAINT fk FOREIGN KEY(dno) REFERENCES 
department(Dnum);


UPDATE employee 
SET dno=4
where eid=1;


UPDATE employee 
SET dno=1
where eid=2;


UPDATE employee 
SET dno=3
where eid=3;


UPDATE employee 
SET dno=3
where eid=4;


UPDATE employee 
SET dno=3
where eid=5;


CREATE TABLE dlocation 
(dno int REFERENCES department(dnum),  
location  VARCHAR(10), 
PRIMARY KEY(dno,location)); 


INSERT INTO dlocation VALUES(1,'Mangaluru');
INSERT INTO dlocation VALUES(1,'bengaluru');
INSERT INTO dlocation VALUES(2,'Mangaluru');
INSERT INTO dlocation VALUES(3,'Mangaluru');
INSERT INTO dlocation VALUES(4,'Mangaluru');


CREATE TABLE project(
Pnum int PRIMARY KEY,
Pname VARCHAR(20), 
Plocation VARCHAR(20), 
dno int  REFERENCES    department(dnum)
);


INSERT INTO project VALUES(1,'Iot','Managluru',1);
INSERT INTO project VALUES(2,'Data Mining','Managluru',1);
INSERT INTO project VALUES(3,'CC','Hubli',3);
INSERT INTO project VALUES(4,'Image processing','Managluru',4);
INSERT INTO project VALUES(5,'Research','Managluru',5);


CREATE TABLE  workson (
Eid  int REFERENCES employee(eid),
Pno  int REFERENCES   project(Pnum),
hours int, 
primary key(Eid,Pno));


INSERT INTO WORKSON  VALUES (1, 1, 4);
INSERT INTO WORKSON  VALUES (2, 1, 5);
INSERT INTO WORKSON  VALUES (3, 2, 4);
INSERT INTO WORKSON  VALUES (4, 3, 4);
INSERT INTO WORKSON  VALUES (5, 5, 4);


CREATE TABLE DEPENDENT(
EMP_ID INT PRIMARY KEY, 
DEPENDENT_NAME VARCHAR(12),
GENDER VARCHAR(5), 
BDATE DATE, 
RELATIONSHIP VARCHAR(12), 
FOREIGN KEY(EMP_ID)REFERENCES EMPLOYEE(EID) ON DELETE CASCADE
); 



-- Query 1

SELECT w.pno 
FROM workson w,employee e
WHERE w.Eid =e.eid 
and e.name='Rahul' 
UNION 
SELECT Pnum 
FROM project p,department d,employee e
WHERE p.dno=d.dnum and d.DMgr_id=e.eid
and  e.name='Rahul';



-- Query 2
SELECT Eid,name, salary,salary+0.1*salary as updated_salary 
FROM employee  
WHERE Eid IN
(SELECT Eid FROM workson WHERE pno IN(
SELECT pnum FROM project WHERE Pname='IOT'));


-- Query 3
SELECT SUM(salary), AVG(salary), MAX(salary), MIN(salary)
FROM employee e,department d
WHERE d.dnum=e.dno AND dname='Account';


-- Query 4

SELECT Name
FROM EMPLOYEE e
WHERE  NOT EXISTS (
    SELECT eid
    from project , workson w
    where pnum =w.pno and w.eid =e.eid 
    and dno not in (select dno from project where dno=5));


-- QUERY 5

Create view deptnew as
select d.dname,count(*), sum(e.salary)
from employee e, department d
where d.dnum=e.dno 
group by dnum;
SELECT * FROM deptnew;



