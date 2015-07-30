package org.acl.deepspark.nn.functions;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.NDArrayUtil;

public class ActivatorFactory {
	public static Activator getActivator(ActivatorType t) {
		switch(t) {
		case SIGMOID:
			return new Activator() {
				
				@Override
				public INDArray output(INDArray input) {
					return Transforms.sigmoid(input);
				}
				
				@Override
				public INDArray derivative(INDArray input, boolean output) {
					if(output) {
						INDArray result = Nd4j.ones(input.shape());
						return result.subi(input).muli(input);
					} else {
						INDArray ret = Nd4j.onesLike(input);
						INDArray f = output(input);
						return ret.subi(f).muli(f);
					}
				}
			};
			
		case RECTIFIED_LINEAR:
			return new Activator() {
				@Override
				public INDArray output(INDArray input) {
					INDArray idx = input.gt(0);
					return input.mul(idx);
				}
				
				@Override
				public INDArray derivative(INDArray input, boolean output) {
					// TODO Auto-generated method stub
					return input.gt(0);
				}
			};
			
		case SOFTMAX: // only for output
			return new Activator() {
				@Override
				public INDArray output(INDArray input) {
					// exp(theta_j^T X) / sum(exp(theta_j^T X))
					INDArray movedInput = input.sub(Nd4j.max(input));
					INDArray output = Transforms.exp(movedInput,true);  
					output.divi(Nd4j.sum(output));
					return output;
				}
				
				@Override
				public INDArray derivative(INDArray input, boolean output) {
					return Nd4j.ones(input.shape());

				}
			};
		case NONE:
			return new Activator() {
				
				@Override
				public INDArray output(INDArray input) {
					// TODO Auto-generated method stub
					return input.dup();
				}
				
				@Override
				public INDArray derivative(INDArray input, boolean output) {
					// TODO Auto-generated method stub
					return Nd4j.ones(input.shape());
				}
			};
		default:
			return null;
		}
		
	}
}
