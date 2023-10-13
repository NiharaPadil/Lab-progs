/*Chef is going for vacations on hill station this winter and is willing to go to the following 
destinations: Manali, Mussoorie, Gulmarg.Further he wants to explore the places by knowing the 
location, the place is famous for and the authentic dishes available in that places. Develop a Java 
program to display the above information based on chef’s destination points.*/




package project7;
import java.util.*;
class HillStation {
 String name, place, village;
 int age;
 Scanner S = new Scanner(System.in);
 void read() {
 System.out.printf("Chef Name: ");
 name = S.next();
 System.out.printf("Place Name: ");
 place = S.next();
 System.out.printf("Village: ");
 village = S.next();
 System.out.printf("Age: ");
 age = S.nextInt();
 }
 void display() {
 System.out.println("Chef Name\tPlace\tVillage\tAge");
 System.out.println(name+"\t"+place+"\t"+village+"\t"+age+"\n");
 }
}
 
class Manali extends HillStation {
 String places[] = new String[20];
 String dishes[] = new String[20];
 int n, m;
 void read() {
 super.read();
 System.out.printf("Enter no. of places & dishes: ");
 n = S.nextInt();
 m = S.nextInt();
 System.out.printf("Enter places name: ");
 for(int i=0; i<n; i++)
 places[i] = S.next();
 System.out.printf("Enter dish name: ");
 for(int i=0; i<m; i++)
 dishes[i] = S.next();
 }
 void display() {
 super.display();
 System.out.println("Places\tDishes");
 for(int i=0; i<n; i++) {
 System.out.println(places[i]);
 }
 for(int i=0; i<n; i++) {
 
 System.out.println("\t\t"+dishes[i]);
 }
 }
 }
 
class Mussoorie extends HillStation {
 String places[] = new String[20];
 String dishes[] = new String[20];
 int n, m;
 void read() {
 super.read();
 System.out.printf("Enter no. of places & dishes: ");
 n = S.nextInt();
 m = S.nextInt();
 System.out.printf("Enter places name: ");
 for(int i=0; i<n; i++)
 places[i] = S.next();
 System.out.printf("Enter dish name: ");
 for(int i=0; i<m; i++)
 dishes[i] = S.next();
 }
 void display() {
 super.display();
 System.out.println("Places\tDishes");
 for(int i=0; i<n; i++) {
 System.out.println(places[i]);
 }
 for(int i=0; i<n; i++) {
 
 System.out.println("\t\t"+dishes[i]);
 }
 }
 }
 
class Gulmarg extends HillStation {
 String places[] = new String[20];
 String dishes[] = new String[20];
 int n, m;
 void read() {
 super.read();
 System.out.printf("Enter no. of places & dishes: ");
 n = S.nextInt();
 m = S.nextInt();
 System.out.printf("Enter places name: ");
 for(int i=0; i<n; i++)
 places[i] = S.next();
 System.out.printf("Enter dish name: ");
 for(int i=0; i<m; i++)
 dishes[i] = S.next();
 }
 void display() {
 super.display();
 System.out.println("Places\tDishes");
 for(int i=0; i<n; i++) {
 System.out.println(places[i]);
 }
 for(int i=0; i<n; i++) {
 
 System.out.println("\t\t"+dishes[i]);
 }
 }
 }
public class Asiign3a {
 public static void main(String[] args) {
 Manali manali = new Manali();
 Mussoorie mussoorie = new Mussoorie();
 Gulmarg gulmarg = new Gulmarg();
 System.out.println("Manali: ");
 manali.read();
 manali.display();
 
 System.out.println("Mussoorie: ");
 mussoorie.read();
 mussoorie.display();
 
 System.out.println("Gulmarg: ");
 gulmarg.read();
 gulmarg.display();
 }
}
