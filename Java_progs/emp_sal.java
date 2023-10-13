/*In the IT company, there are 20 employees and the monthly salary of employees must be
arranged in ascending order. Write a Java program to read 20 employees’ salaries as floatingpoint numbers and apply for each to perform the sorting, and display the result. Also, the
industry places the highest paid employee in the first level, the next highest paid two
employees placed in the second level and next three highest paid employees in the third level,
and so on.*/ 




import java.util.*;
import java.util.Arrays;
import java.util.Scanner;
class Assignm1b
{
public static void main(String args[])
{
float employee_salary[]=new float[21];
float temp;
Scanner sc=new Scanner(System.in);
System.out.println("ENter the salaries of 21 employees");
for(int i=0;i<21;i++)
{
employee_salary[i]=sc.nextFloat();
}
Arrays.sort(employee_salary);
 System.out.println("Sorted Array: ");
 for (float arr1 : employee_salary) {
 System.out.println(arr1);
 }
 for (int i = 0; i < employee_salary.length; i++) { 
 for (int j = i+1; j < employee_salary.length; j++) { 
 if(employee_salary[i] < employee_salary[j]) { 
 temp = employee_salary[i]; 
 employee_salary[i] = employee_salary[j]; 
 employee_salary[j] = temp; 
 } 
 } 
 } 
 
 System.out.println(); 
 System.out.println("Elements of array sorted in descending order: "); 
 for (int i = 0; i < employee_salary.length; i++) { 
 System.out.println(employee_salary[i] + " "); 
 } 
 System.out.println(); 
 System.out.println("information 21 employees as below:");
 int k=0;
 float y[][]=new float[6][6];
 for(int i=0; i<6; i++)
 { 
 for(int j=0; j<=i; j++)
 {
 y[i][j]=employee_salary[k];
 System.out.print(y[i][j]+" ");
 k++;
 }
 System.out.println();
 }
 }
}
