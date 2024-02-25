import java.io.IOException;
import java.net.*;
public class udps {
    public static void main(String[] args) {
        try{
            DatagramSocket serverSocket = new DatagramSocket(9876);
            byte[] receivedData = new byte[1024];
            byte[] sendData = "hey".getBytes();
            DatagramPacket receivedPacket = new DatagramPacket(receivedData, receivedData.length);
            System.out.println("Server started, waiting for clients");
            serverSocket.receive(receivedPacket);
            String receivedMessage = new String(receivedPacket.getData(),0,receivedPacket.getLength());
            InetAddress clientIpAdrress =receivedPacket.getAddress();
            int clientPort = receivedPacket.getPort();
            DatagramPacket sendpacket =  new DatagramPacket(sendData,sendData.length,clientIpAdrress,clientPort);
            serverSocket.send(sendpacket);
            serverSocket.close();
        }catch(IOException e){
            e.printStackTrace();
        }
    }
}
