/*Create a java program to generate HallTicket with the methods FeesDues, 
AttendenceChecklist and RegularBacklogSub using packages.*/


package project4b;
import P2.hallTicket;
import java.util.Scanner;
public class Assign4b {
 public static void main(String[] args) {
 Scanner S = new Scanner(System.in);
 System.out.println(".......... Generate Hallticket ..........");
 HallTicket t = new HallTicket();
 t.feeDue();
 t.AttendanceChecklist();
 t.RegularBacklogSub();
 t.HallTicket();
 
 }
 
 }
 
 }
}
package P2;
import java.util.Scanner;
class HallTicket {
 Scanner S = new Scanner(System.in);
 String Name, USN, Branch;
 int noSub;
 int Sem;
 long feePaid, fee=5000,rem_amount;
 float atdPass = 85, attend, total;
 String subjects[] = new String[10];
 int backlogSub;
 
 public HallTicket() {
 System.out.print("Enter your name: ");
 Name = S.next();
 System.out.print("Enter your USN: ");
 USN = S.next();
 System.out.print("Enter your branch: ");
 stdBranch = S.next();
 System.out.print("Enter your semester: ");
 Sem = S.nextInt();
 System.out.print("Enter your fee paid: ");
 feePaid = S.nextLong();
 System.out.print("Enter number of subjects: ");
 noSub = S.nextInt();
 System.out.println("Enter code of the subjects: ");
 for(int i=0; i<noSub; i++) {
 System.out.print((i+1)+". ");
 subjects[i]=S.next();
 }
 }
 
 void feeDue() {
 if(feePaid < fee){
 rem_amount=fee-feePaid;
 System.out.println("feedues="+rem_amount);
 else
 System.out.println(“no feedues”);
 }
 }
 
 public void RegularBacklogSub() {
 System.out.println("Enter number of backlogs: ");
 backlogSub = S.nextInt();
 if (backlogSub>0) {
 System.out.println("There is backlogs.");
 else
 System.out.println(“no backlogs”);
 }
 }
 
 public void AttendanceChecklist() {
 for(int i=0; i<noSub; i++) {
 System.out.print("Enter total number of classes for "+subjects[i]+": ");
 total = S.nextFloat();
 System.out.print("Enter total number of classes attended for "+subjects[i]+": ");
 attend = S.nextFloat();
 }
 for(int i=0; i<noSub; i++) {
 if (((attend/total)*100)<atdPass) {
 percentage=(attend/total)*100;
 System.out.print("There is shortage of attendance."+percentage);
 else
 System.out.println(“no shortage”);
 }
 }
 }
 
 public void HallTicket() {
System.out.println("=============== Hall Ticket ==============================");
System.out.println("Name: "+stdName+ "\nUSN: "+stdUSN+"\nBranch: "+stdBranch+"\nSemester:
"+stdSem);
 System.out.println("Subjects:");
 for(int i=0; i<noSub; i++) {
 System.out.println(subjects[i]+" ");
 
}
 
}
}