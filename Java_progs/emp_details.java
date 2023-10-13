/*Create a Java class called Employee with the following details as variables within it. (i)
Empid(ii)Emp_ Name (iii) Salary (iv) Phone_no. Write a Java program to create nEmployee
objects and print the Empid, Emp_Name, Salary, and Phone_no of these objects with suitable
headings.*/




import java.util.Scanner;
class Employee
{
String Eid;
String name;
double Salary;
long phone;
public void getInfo()
{
Scanner in=new Scanner(System.in);
System.out.println("Enter the id:");
Eid=in.next();
System.out.println("Enter the Name:");
name=in.next();
System.out.println("Enter the salary:");
Salary=in.nextDouble();
System.out.println("Enter the Phone No:");
phone=in.nextLong();
}
public void display()
{
System.out.println(Eid+"\t"+name+"\t"+Salary+"\t\t"+phone);
}
}
public class Assign2c
{
public static void main(String[] args)
{
int i;
Employee[] e=new Employee[100];
Scanner in=new Scanner(System.in);
System.out.println("Enter the number of Employees");
int n=in.nextInt();
System.out.println("Enter the Employee details");
for(i=0;i<n;i++)
{
e[i]=new Employee();
e[i].getInfo();
}
System.out.println("Id\t\tName\tSalary\tPhone");
System.out.println("........................................……………………………….");
for(i=0;i<n;i++)
{
e[i].display();
}
}
}