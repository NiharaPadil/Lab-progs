/*There is a Vehicle class that encapsulates information about vehicles, including the vehicle 
number, their fuel capacity, and their fuel consumption rate. Use the Vehicle class as a starting point
from which more specialized classes are developed. For example, one type of vehicle is a Truck to 
carry its cargo capacity. Thus, to create a Truck class extend the Vehicle class, adding an instance 
variable that stores the goods carrying capacity. One more type of vehicle is Bus to carry people 
with a seating capacity. Create Bus class extends vehicle class, adding an instance of the seating 
capacity. Use the appropriate access specifier for data members and the methods are provided to 
read and display their values.*/



class Vehicle
{
String vehicle_id;
int fuel_cap;
int fuel_cons_rate;
Vehicle(String vid, int fc, int fcr)
{
vehicle_id=vid;
fuel_cap=fc;
fuel_cons_rate=fcr;
}
void display()
{
System.out.println("");
System.out.print("vehicle_id fuel_cap fuel_cons_rate Seating_Capacity Carrying_capacity");
System.out.print("\n");
System.out.println(vehicle_id+"\t"+fuel_cap+"\t"+fuel_cons_rate+"\t");
}
}
class Bus extends Vehicle
{
int seating_cap;
Bus(String vid, int fc, int fcr,int ss)
{
super(vid,fc,fcr);
seating_cap=ss;
}
void display()
{
System.out.println(".........Bus Information.............");
super.display();
System.out.println(seating_cap+"\t****");
}
}
class Truck extends Vehicle
{
int carrying_cap;
Truck(String vid, int fc, int fcr,int cc)
{
super(vid,fc,fcr);
carrying_cap=cc;
}
void display()
{
System.out.println("..........Truck Information............");
super.display();
System.out.println("\t****\t"+carrying_cap);
}
}
class Assign3b
{
public static void main(String args[])
{
Bus b=new Bus("1234",60,25,50);
Truck t=new Truck("1236",50,20,40);
b.display();
System.out.println();
t.display();
System.out.println();
}
}