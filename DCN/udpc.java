import java.io.IOException;
import java.net.*;
class udpc{
    public static void main(String[] args) {
        try{
            DatagramSocket clienSocket = new DatagramSocket();
            InetAddress IPAddress = InetAddress.getByName("localhost");
            byte[] sendData="hi".getBytes();
            DatagramPacket sendPacket = new DatagramPacket(sendData, sendData.length,IPAddress,9876);
            clienSocket.send(sendPacket);
            byte[] receivedData = new byte[1024];
            DatagramPacket receivePacket = new DatagramPacket(receivedData, receivedData.length);
            clienSocket.receive(receivePacket);
            String message= new String(receivePacket.getData(),0,receivePacket.getLength());
            System.out.println("Received from server "+message);
            clienSocket.close();
        }catch(IOException e){
            e.printStackTrace();
        }
    }
}