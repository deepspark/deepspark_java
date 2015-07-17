package org.acl.deepspark.nn.functions;

import org.nd4j.linalg.api.ndarray.INDArray;

public class ActivatorFactory {
	public static Activator getActivator(ActivatorType t) {
		switch(t) {
		case SIGMOID:
			return new Activator() {
				
				@Override
				public INDArray output(INDArray input) {
					
					return null;
				}
				
				@Override
				public INDArray derivative(INDArray input) {
					// TODO Auto-generated method stub
					return null;
				}
			};
		case RECTIFIED_LINEAR:
			return new Activator() {
				@Override
				public INDArray output(INDArray input) {
					// TODO Auto-generated method stub
					return null;
				}
				
				@Override
				public INDArray derivative(INDArray input) {
					// TODO Auto-generated method stub
					return null;
				}
			};
		case SOFTMAX:
			return new Activator() {
				@Override
				public INDArray output(INDArray input) {
					// TODO Auto-generated method stub
					return null;
				}
				
				@Override
				public INDArray derivative(INDArray input) {
					// TODO Auto-generated method stub
					return null;
				}
			};
		default:
			return null;
		}
		
	}
}
