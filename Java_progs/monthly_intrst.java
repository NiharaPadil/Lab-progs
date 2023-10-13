/*In a financial application computing the CD value, where CD is the certificate of deposit
allows you to save money at a fixed interest rate for a fixed amount of time. Suppose you put
INR10,000 into a CD with an annual percentage yield of 5.75%. After one month, the CD is
worth INR10,047.91 calculated as 10000+10000*(5.75/12*100). Write a program using
appropriate class and objects that prompts the user to enter an amount, the annual
percentage yield*/



import java.util.*;
class Compund
{
double monthlyInterestRate;
double cdValue;
double deposit;
double apy;
int maturityPeriod;
void Compute()
{
System.out.print("Enter the initial deposit amount: ");
Scanner input =new Scanner(System.in);
deposit = input.nextDouble();
System.out.print("Enter anual percentage yield: ");
apy = input.nextDouble();
System.out.print("Enter maturity period (number of months): ");
maturityPeriod = input.nextInt();
System.out.println("Month" + "\t" + "CD Value");
for (int i = 1; i <= maturityPeriod; i++){
deposit += deposit * (apy / 1200);
System.out.println(i + "\t" + deposit );
}
}
}
public class Assign2b {
public static void main(String[] args) {
Compund Ci=new Compund();
Ci.Compute();
System.out.print("");
}
}
