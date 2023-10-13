/*Develop a Java program to create a class LIFT with members person_no, max_capacity,
no_of_floors and fire_alarm. Write the zero parameter and parametrized constructors to create the
LIFT objects with lifts information. Create methods AddPersons and ExitPersons to add and remove
the persons from the lift. Throw a user defined exception "PowerOffException” with the message
"Lift cannot be operated now" when there is no power. Also create and throw a user defined
exception "LiftFullException" when the total capacity of people is more than max_capacity with the
message "Lift is almost full. You cannot enter". The Program should also throw an exception
"FireAlarmException" when there is a fire in the lift.*/





import java.util.*;
class PowerOffException extends Exception{
 PowerOffException(String s){
 super(s);
 }
}
class LiftFullException extends Exception{
 LiftFullException (String s){
 super(s);
 }
}
class FireAlarmException extends Exception{
 FireAlarmException (String s){
 super(s);
}
}
class lift{
 int person_no,max_capacity,no_of_floors;
 int firealarm,power;
 
 
 
 
 lift(int person_no,int max_capacity,int no_of_floors, int firealarm,int power){
 this.person_no=person_no;
 this.max_capacity=max_capacity;
 this.no_of_floors=no_of_floors;
 this.firealarm=firealarm;
 this.power=power;
 }
 void addperson(int num)
 
 
 
 throws PowerOffException,LiftFullException,FireAlarmException{
 
 
 if(power==0){
 throw new PowerOffException("Lift cannot be operated now since power is off");
 }
 if(firealarm==1)
 {
 throw new 
 FireAlarmException("Fire alarm is ON you cannot enter the lift");
 }
 if(person_no+num>max_capacity){
 throw new
 LiftFullException("Lift is almost full. You cannot enter");
 }
 person_no+=num;
 }
 void ExitPerson(int num1)throws
 PowerOffException,LiftFullException{
 
 if(power==0){
 throw new
 PowerOffException("Lift cannot be operated now since power is off");
 }
 if(person_no-num1<=max_capacity)
 {
 throw new
 LiftFullException("lift can be used");
 }
 person_no-=num1;
 }
 }
 
 public class Assign7a {
 public static void main(String[] args) {
 int num,num1;
 Scanner sc=new Scanner(System.in);
 System.out.println("WELCOME TO THE LIFT");
 System.out.println("Enter the number of person inside the lift");
 int person_no=sc.nextInt();
 System.out.println("Enter the max capacity of the lift");
 int max_capacity=sc.nextInt();
 System.out.println("Enter the number of floors");
 int no_of_floors=sc.nextInt();
 System.out.println("firealarm is ON or OFF press 1 for ON and press 0 for OFF");
 int firealarm=sc.nextInt();
 System.out.println("Does the power is there or not PRESS 1 for ON and 0 for OFF");
 int power=sc.nextInt();
 lift L=new lift(person_no,max_capacity,no_of_floors,firealarm,power);
 try{
 
 System.out.println("Enter the no of people want to enter the lift");
 num=sc.nextInt();
 L.addperson(num);
 
 System.out.println("Enter the no of people exited");
 num1=sc.nextInt();
 L.ExitPerson(num1);
 
 }
 catch(Exception PowerOffException){
 System.out.println(PowerOffException.getMessage());
 }
 
 
 
 }
 
}