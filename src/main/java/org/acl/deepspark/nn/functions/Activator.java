package org.acl.deepspark.nn.functions;

public class Activator {
	public static double sigmoid(double x) {
		double temp = Math.exp(-x);
		return (1 - temp) / (1 + temp);
	}
}
