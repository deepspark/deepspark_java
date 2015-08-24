package org.acl.deepspark.nn.async;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;

import org.acl.deepspark.data.Weight;

public class ParameterClient {
	public static void sendDelta(String host, int port, Weight[] d) throws IOException, ClassNotFoundException {
		Socket s = new Socket(host, port);
		ObjectOutputStream os = new ObjectOutputStream(s.getOutputStream());
		os.writeObject(d);
		os.flush();
		s.close();
	}
	
	public static Weight[] getWeights(String host, int port) throws IOException, ClassNotFoundException {
		Weight[] w = null;
		
		Socket s = new Socket(host, port);
		ObjectInputStream os = new ObjectInputStream(s.getInputStream());
		w = (Weight[]) os.readObject();
		s.close();
		
		return w;
	}
}
