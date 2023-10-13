/*Develop a Java program to create a class Fruits with members fruitname and qty. Write the zero
parameter and parametrized constructors to create the fruits objects with fruits information. Create
methods Addfruits and Usefruits to add and use the fruits from the existing quantity. Throw a user
defined exception "NotEnoughFruits” with the message "You have only xx number of fruits" when
quantity of used fruits less than existing quantity. Create and throw a user defined exception
"BoxFullException" when the total quantity of fruits is more than 100 with the message "Box is
almost full. You can add only xx fruits". The Program should also throw an exception "Lowstock"
when the quantity becomes zero. Demonstrate the use of throws clause.*/


import java.util.*;
class NotEnoughFruits extends Exception {
 NotEnoughFruits(String s){
 super(s);
 }
}
class BoxFullException extends Exception {
 BoxFullException(String s){
 super(s);
 }
}
class LowStockException extends Exception {
 LowStockException(String s){
 super(s);
 }
}
class Fruits {
 String fruitname;
 int qty;
 // zero parameter constructor
 Fruits() {
 fruitname = "apple";
 qty = 0;
 }
 // parametrized constructor
 Fruits(String fruitname, int qty) {
 this.fruitname = fruitname;
 this.qty = qty;
 }
 // method to add fruits
 void AddFruits(int num) throws BoxFullException {
 if (qty + num > 100) {
 throw new BoxFullException("Box is almost full. You can add only " + (100-qty) + "
fruits.");
 }
 qty += num;
 }
 // method to use fruits
 void UseFruits(int num) throws NotEnoughFruits, LowStockException {
 if (qty - num < 0) {
 throw new NotEnoughFruits("You have only " + qty + " number of fruits.");
 }
 if (qty - num == 0) {
 throw new LowStockException("Stock is low, please add more fruits.");
 }
 qty -= num;
 }
}
 public class Assign7b {
 public static void main(String[] args) {
 int num,num1;
 Scanner sc=new Scanner(System.in);
 
 Fruits L=new Fruits("Mango",70);
 try{
 
 System.out.println("Enter the no of fruits to be placed in the box");
 num=sc.nextInt();
 L.AddFruits(num);
 
 System.out.println("Enter the no of fruits removed from the box");
 num1=sc.nextInt();
 L.UseFruits(num1);
 
 }
 catch(Exception PowerOffException){
 System.out.println(PowerOffException.getMessage());
 }
 
 
 
 }
 
}
7c) Develop a Java program for ArithmeticException and ArrayOutOfBoundsException shows the
arithmetic error and array index error.
The program includes
(i) The input instances of type double.
(ii) Methods to read input.
(iii) Application of the try and catch methods.
import java.util.Scanner;
public class Assign7c {
 public static void main(String[] args) {
 Scanner sc = new Scanner(System.in);
 // input instances
 double num1, num2;
 // reading input
 System.out.print("Enter first number: ");
 num1 = sc.nextDouble();
 System.out.print("Enter second number: ");
 num2 = sc.nextDouble();
 // try-catch block for arithmetic exception
 try {
 if(num2==0)
 throw new ArithmeticException("Devide by 0 error");
 result = num1 / num2;
 System.out.println("Result: " + result);
 } 
 catch (ArithmeticException e)
 {
 System.out.println("Error: Cannot divide by zero.");
 }
 // input instances
 int size;
 int[] arr;
 // reading input
 System.out.print("Enter size of array: ");
 size = sc.nextInt();
 arr = new int[size];
 System.out.println("Enter elements of array: ");
 for (int i = 0; i < size; i++) {
 arr[i] = sc.nextInt();
 }
 // try-catch block for array out of bounds exception
 try {
 int index;
 System.out.print("Enter index to access: ");
 index = sc.nextInt();
 System.out.println("Element at index " + index + ": " + arr[index]);
 } catch (ArrayIndexOutOfBoundsException e) {
 System.out.println("Error: Invalid index.");
 }
 }
}