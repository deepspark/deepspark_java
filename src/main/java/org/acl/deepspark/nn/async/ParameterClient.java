package org.acl.deepspark.nn.async;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.Socket;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import org.acl.deepspark.data.Weight;

public class ParameterClient {
	public static void sendDelta(String host, int port, Weight[] d) throws IOException, ClassNotFoundException {
		Socket s = new Socket(host, port);
		s.setSoTimeout(15000);
		ObjectOutputStream os = new ObjectOutputStream(new GZIPOutputStream(s.getOutputStream()));
		os.writeObject(d);
		s.close();
	}
	
	public static Weight[] getWeights(String host, int port) throws IOException, ClassNotFoundException {
		Socket s = new Socket(host, port);
		s.setSoTimeout(15000);
		ObjectInputStream os = new ObjectInputStream(new GZIPInputStream(s.getInputStream()));
		Weight[] w = (Weight[]) os.readObject();
		s.close();
		
		return w;
	}
}
