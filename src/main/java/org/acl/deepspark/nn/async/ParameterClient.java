package org.acl.deepspark.nn.async;

import org.acl.deepspark.data.Weight;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;

public class ParameterClient {
	public static void sendDelta(String host, int port, Weight[] d) throws IOException, ClassNotFoundException {
		Socket s = new Socket(host, port);
		s.setSoTimeout(15000);
		ObjectOutputStream os = new ObjectOutputStream(s.getOutputStream());
		os.writeObject(d);
		s.close();
	}
	
	public static Weight[] getWeights(String host, int port) throws IOException, ClassNotFoundException {
		Socket s = new Socket(host, port);
		s.setSoTimeout(15000);
		ObjectInputStream os = new ObjectInputStream(s.getInputStream());
		Weight[] w = (Weight[]) os.readObject();
		s.close();
		
		return w;
	}
}
