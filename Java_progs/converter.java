/*Develop a Java application to implement currency converter (Dollar to INR, EURO to 
INR, Yen to INR and vice versa), distance converter (meter to KM, miles to KM and vice 
versa), time converter (hours to minutes, seconds and vice versa) using packages.*/


//CurrencyConverter


package cc;
import java.util.*;
public class CurrencyC
{
double inr,usd;
double euro,yen;
Scanner in=new Scanner(System.in);
public void dollartorupee()
{
System.out.println("Enter dollars to convert into Rupees:");
usd=in.nextInt();
inr=usd*81.83;
System.out.println("Dollar ="+usd+" equal to INR="+inr);
System.out.println("\n");
}
public void rupeetodollar()
{
System.out.println("Enter Rupee to convert into Dollars:");
inr=in.nextInt();
usd=inr/81.83;
System.out.println("Rupee ="+inr+"equal to Dollars="+usd);
}
public void eurotorupee()
{
System.out.println("Enter Euro to convert into Rupees:");
euro=in.nextInt();
inr=euro*79.06;
System.out.println("Euro ="+euro+" equal to INR="+inr);
System.out.println("\n");
}
public void rupeetoeuro()
{
System.out.println("Enter Rupees to convert into Euro:");
inr=in.nextInt();
euro=(inr/79.06);
System.out.println("Rupee ="+inr +"equal to Euro="+euro);
System.out.println("\n");
}
public void yentoruppe()
{
System.out.println("Enter Yen to convert into Rupees:");
yen=in.nextInt();
inr=yen*0.57;
System.out.println("Yen ="+yen+" equal to INR="+inr);
System.out.println("\n");
}
public void ruppetoyen()
{
System.out.println("Enter Rupees to convert into Yen:");
inr=in.nextInt();
yen=(inr/0.57);
System.out.println("INR="+inr +"equal to YEN"+yen);
System.out.println("\n");
}
}

//Distance converter:


package distanceconversion;
import java.util.*;
public class Distance
{
double km,m,miles;
Scanner sc = new Scanner(System.in);
public void kmtom()
{
System.out.print("Enter in km ");
km=sc.nextDouble();
m=(km*1000);
System.out.println(km+"km" +"equal to"+m+"metres");
}
public void mtokm()
{
System.out.print("Enter in meter ");
m=sc.nextDouble();
km=(m/1000);
System.out.println(m+"m" +"equal to"+km+"kilometres");
}
public void milestokm()
{ 
System.out.print("Enter in miles");
miles=sc.nextDouble();
km=(miles*1.60934);
System.out.println(miles+"miles" +"equal to"+km+"kilometres");
}
public void kmtomiles()
{
System.out.print("Enter in km");
km=sc.nextDouble();
miles=(km*0.621371);
System.out.println(km+"km" +"equal to"+miles+"miles");
}
}


//Timer


package timeconversion;
import java.util.*;
public class Timer
{
int hours,seconds,minutes;
int input;
Scanner sc = new Scanner(System.in);
public void secondstohours()
{
System.out.print("Enter the number of seconds: ");
input = sc.nextInt();
hours = input / 3600;
minutes = (input % 3600) / 60;
seconds = (input % 3600) % 60;
System.out.println("Hours: " + hours);
System.out.println("Minutes: " + minutes);
System.out.println("Seconds: " + seconds);
}
public void minutestohours()
{
System.out.print("Enter the number of minutes: ");
minutes=sc.nextInt();
hours=minutes/60;
minutes=minutes%60;
System.out.println("Hours: " + hours);
System.out.println("Minutes: " + minutes);
}
public void hourstominutes()
{ 
System.out.println("enter the no of hours");
hours=sc.nextInt();
minutes=(hours*60);
System.out.println("Minutes: " + minutes);
}
public void hourstoseconds()
{ 
System.out.println("enter the no of hours");
hours=sc.nextInt();
seconds=(hours*3600);
System.out.println("Minutes: " + seconds);
}
}
Main method:
import java.util.*;
import java.io.*;
import cc.*;
import distanceconversion.*;
import timeconversion.*;
class Converter
{
public static void main(String args[])
{
Scanner s=new Scanner(System.in);
int choice,ch;
CurrencyC c=new CurrencyC();
Distance d=new Distance();
Timer t=new Timer();
do
{
System.out.println("1.dollar to rupee ");
System.out.println("2.rupee to dollar ");
System.out.println("3.Euro to rupee ");
System.out.println("4..rupee to Euro ");
System.out.println("5.Yen to rupee ");
System.out.println("6.Rupee to Yen ");
System.out.println("7.Meter to kilometer ");
System.out.println("8.kilometer to meter ");
System.out.println("9.Miles to kilometer ");
System.out.println("10.kilometer to miles");
System.out.println("11.Hours to Minutes");
System.out.println("12.Hours to Seconds");
System.out.println("13.Seconds to Hours");
System.out.println("14.Minutes to Hours");
System.out.println("Enter ur choice");
choice=s.nextInt();
switch(choice)
{
case 1:
{
c.dollartorupee();
break;
}
case 2:
{
c.rupeetodollar();
break;
}
case 3:
{
c.eurotorupee();
break;
}
case 4:
{
c.rupeetoeuro();
break;
}
case 5:
{c.yentorupee();
break;}
case 6 :
{
c.rupeetoyen();
break;
}
case 7 :
{
d.mtokm();
break;
}
case 8 :
{
d.kmtom();
break;
}
case 9 :
{
d.milestokm();
break;
}
case 10 :
{
d.kmtomiles();
break;
}
case 11 :
{
t.hourstominutes();
break;
}
case 12 :
{
t.hourstoseconds();
break;
}
case 13 :
{
t.secondstohours();
break;
}
case 14 :
{
t.minutestohours();
break;
}}
System.out.println("Enter 0 to quit and 1 to continue ");
ch=s.nextInt();
}while(ch==1);
}
}
