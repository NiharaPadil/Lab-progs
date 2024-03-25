-- Consider the following schema for Bank Database:
-- BRANCH(Branch_id, Bank_name, Branch_name, Assets)
-- ACCOUNT(Acc_no, Branch_id, Account_Type, Account_Balance, Customer_id) CUSTOMER(Customer_id, Customer_name, Customer_age, Customer_address, Customer_phone) LOAN(Loan_number, Branch_id, Amount, Customer_id)
-- Note: Account_Type may be of following: Savings, Recurrent, Fixed Deposit

-- Write SQL queries to
-- 1. Find all the Customers who have at least one account at the “Mangaluru”  branch.
-- 2. Find names of the depositors who have deposited highest amount among all the customers.
-- 3. Retrieve the Customer name and loan amount of a customer who borrowed a loan more than
-- 5,00,000.
-- 4. Retrieve  the  details  of  bank  branch  with  maximum  and  minimum  assets  among  the  various branches.
-- 5. Demonstrate how you delete all account tuples at every branch located in a specific city. 

create schema Bank;
use bank;

CREATE TABLE BRANCH(
BRANCH_ID VARCHAR(25) PRIMARY KEY,
BANK_NAME VARCHAR(25),
BRANCH_NAME VARCHAR(25),
ASSETS VARCHAR(25) 
);

CREATE TABLE CUSTOMER(
CUSTOMER_ID VARCHAR(25) PRIMARY KEY,
CUSTOMER_NAME VARCHAR(25),
CUSTOMER_AGE VARCHAR(25),
CUSTOMER_ADDRESS VARCHAR(25),
CUSTOMER_PHONE VARCHAR(25)
);

CREATE TABLE ACCOUNT(
ACC_NO VARCHAR(25) PRIMARY KEY,
BRANCH_ID VARCHAR(25),
ACCOUNT_TYPE VARCHAR(25),
ACCOUNT_BALANCE VARCHAR(25),
CUSTOMER_ID VARCHAR(25),
FOREIGN KEY (BRANCH_ID) REFERENCES BRANCH(BRANCH_ID) ON DELETE CASCADE,
FOREIGN KEY(CUSTOMER_ID) REFERENCES CUSTOMER(CUSTOMER_ID) ON DELETE CASCADE
);

CREATE TABLE LOAN(
LOAN_NUMBER VARCHAR(25) PRIMARY KEY,
BRANCH_ID VARCHAR(25),
AMOUNT VARCHAR(25),
CUSTOMER_ID VARCHAR(25),
FOREIGN KEY (BRANCH_ID) REFERENCES BRANCH(BRANCH_ID) ON DELETE CASCADE,
FOREIGN KEY(CUSTOMER_ID) REFERENCES CUSTOMER(CUSTOMER_ID) ON DELETE CASCADE);


-- Insert values into the BRANCH table
INSERT INTO BRANCH (BRANCH_ID, BANK_NAME, BRANCH_NAME, ASSETS) 
VALUES 
('B001', 'Bank of Mangaluru', 'Mangaluru ', '1000000'),
('B002', 'Bank of Mangaluru', 'Bengaluru ', '500000'),
('B003', 'Bank of Bengaluru', 'Bengaluru ', '750000'),
('B004', 'Bank of Bengaluru', 'Mangaluru ', '600000'),
('B005', 'Bank of Hyderabad', 'Hyderabad ', '400000');

-- Insert values into the CUSTOMER table
INSERT INTO CUSTOMER (CUSTOMER_ID, CUSTOMER_NAME, CUSTOMER_AGE, CUSTOMER_ADDRESS, CUSTOMER_PHONE) 
VALUES 
('C001', 'Alice', '30', '123 Maple Street', '1234567890'),
('C002', 'Bob', '35', '456 Oak Avenue', '2345678901'),
('C003', 'Charlie', '40', '789 Pine Street', '345-678-9012'),
('C004', 'Diana', '45', '987 Elm Street', '4567890123'),
('C005', 'Evan', '50', '654 Cedar Road', '5678901234');

-- Insert values into the ACCOUNT table
INSERT INTO ACCOUNT (ACC_NO, BRANCH_ID, ACCOUNT_TYPE, ACCOUNT_BALANCE, CUSTOMER_ID) 
VALUES 
('101', 'B001', 'Savings', '10000', 'C001'),
('102', 'B002', 'Recurrent', '15000', 'C002'),
('103', 'B003', 'Fixed Deposit', '20000', 'C003'),
('104', 'B004', 'Savings', '25000', 'C004'),
('105', 'B005', 'Recurrent', '30000', 'C005');

-- Insert values into the LOAN table
INSERT INTO LOAN (LOAN_NUMBER, BRANCH_ID, AMOUNT, CUSTOMER_ID) 
VALUES 
('L001', 'B001', '500000', 'C001'),
('L002', 'B002', '450000', 'C002'),
('L003', 'B003', '600000', 'C003'),
('L004', 'B004', '550000', 'C004'),
('L005', 'B005', '700000', 'C005');


-- QUERY 1

SELECT DISTINCT c.*
FROM customer c, branch b, account a
WHERE  b.branch_id = a.branch_id
AND a.customer_id = c.customer_id 
and b.branch_name = 'Mangaluru';

SELECT DISTINCT c.customer_name
FROM customer c
JOIN account a ON c.customer_id = a.customer_id
JOIN branch b ON a.branch_id = b.branch_id
WHERE b.branch_name = 'Mangaluru';


-- QUERY 2

SELECT C.CUSTOMER_NAME 
FROM CUSTOMER C , ACCOUNT A 
WHERE C.CUSTOMER_ID=A.CUSTOMER_ID 
AND ACCOUNT_BALANCE =(SELECT MAX(ACCOUNT_BALANCE) FROM ACCOUNT);



-- QUERY 3

SELECT C.CUSTOMER_NAME,L.AMOUNT
FROM ACCOUNT A , CUSTOMER C, LOAN l
WHERE l.customer_id=c.customer_id and a.customer_id=c.customer_id
and l.amount>500000;

-- 	Query 4
SELECT *
FROM BRANCH 
WHERE ASSETS=(SELECT MAX(ASSETS) FROM BRANCH)
UNION 
SELECT *
FROM BRANCH 
WHERE ASSETS=(SELECT MIN(ASSETS) FROM BRANCH);

-- QUERY 5
DELETE FROM ACCOUNT
WHERE BRANCH_ID IN (
    SELECT BRANCH_ID
    FROM BRANCH
    WHERE BRANCH_NAME = '&BRANCH_NAME'
);
