package org.acl.deepspark.nn.conf.spark;

import org.acl.deepspark.data.Sample;
import org.apache.spark.api.java.function.Function;
import org.jblas.DoubleMatrix;

public class OutputFunction implements Function<Sample, DoubleMatrix> {
	/**
	 * 
	 */
	private static final long serialVersionUID = -6493839448747660091L;
	
	private DistNeuralNetConfiguration nnc;
	
	public OutputFunction(DistNeuralNetConfiguration nnc) {
		this.nnc = nnc;
	}
	
	@Override
	public DoubleMatrix call(Sample arg0) throws Exception {
		return nnc.getOutput(arg0.data)[0].sub(arg0.label);
	}
}


