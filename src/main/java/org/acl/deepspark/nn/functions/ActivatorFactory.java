package org.acl.deepspark.nn.functions;

import org.acl.deepspark.data.Tensor;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

import java.io.Serializable;

public class ActivatorFactory implements Serializable {
	public static Activator get(ActivatorType t) {
		switch(t) {
			case SIGMOID:
				return new Activator() {

					@Override
					public FloatMatrix output(FloatMatrix input) {
						if (input != null) {
							FloatMatrix denom = MatrixFunctions.exp(input.mul(-1)).add(1);
							return FloatMatrix.ones(input.rows, input.columns).divi(denom);
						}
						return null;
					}

					@Override
					public FloatMatrix derivative(FloatMatrix activated) {
						if (activated != null) {
							FloatMatrix ret = FloatMatrix.ones(activated.rows, activated.columns);
							return ret.subi(activated).muli(activated);
						}
						return null;
					}

					@Override
					public Tensor output(Tensor input) {
						if (input != null) {
							int length = input.data().length;
							Tensor ret = Tensor.zeros(input.shape());
							for (int i = 0 ; i < length; i++) {
								ret.data()[i].addi(output(input.data()[i]));
							}
							return ret;
						}
						return null;
					}

					@Override
					public Tensor derivative(Tensor activated) {
						if (activated != null) {
							int length = activated.data().length;
							Tensor ret = Tensor.zeros(activated.shape());
							for (int i = 0 ; i <length; i++) {
								ret.data()[i].addi(derivative(activated.data()[i]));
							}
							return ret;
						}
						return null;
					}
				};
			
			case RECTIFIED_LINEAR:
				return new Activator() {
					@Override
					public FloatMatrix output(FloatMatrix input) {
						if (input != null) {
							FloatMatrix idx = input.gt(0.0f);
							return input.mul(idx);
						}
						return null;
					}

					@Override
					public FloatMatrix derivative(FloatMatrix activated) {
						if (activated != null)
							return activated.gt(0.0f);
						return null;
					}

					@Override
					public Tensor output(Tensor input) {
						if (input != null) {
							int length = input.data().length;
							Tensor ret = Tensor.zeros(input.shape());
							for (int i = 0 ; i < length; i++) {
								ret.data()[i].addi(output(input.data()[i]));
							}
							return ret;
						}
						return null;
					}

					@Override
					public Tensor derivative(Tensor activated) {
						if (activated != null) {
							int length = activated.data().length;
							Tensor ret = Tensor.zeros(activated.shape());
							for (int i = 0 ; i < length; i++) {
								ret.data()[i].addi(derivative(activated.data()[i]));
							}
							return ret;
						}
						return null;
					}
				};

			case SOFTMAX: // only for output
				return new Activator() {
					@Override
					public FloatMatrix output(FloatMatrix input) {
						// exp(theta_j^T X) / sum(exp(theta_j^T X))
						//FloatMatrix movedInput = input.sub(input.max());
						if (input != null) {
							FloatMatrix output = MatrixFunctions.exp(input);
							output.divi(output.sum());
							return output;
						}
						return null;
					}

					@Override
					public FloatMatrix derivative(FloatMatrix activated) {
						if (activated != null)
							return FloatMatrix.ones(activated.rows, activated.columns);
						return null;
					}

					@Override
					public Tensor output(Tensor input) {
						if (input != null) {
							int length = input.data().length;
							Tensor ret = Tensor.zeros(input.shape());
							for (int i = 0 ; i < length; i++) {
								ret.data()[i].addi(output(input.data()[i]));
							}
							return ret;
						}
						return null;
					}

					@Override
					public Tensor derivative(Tensor activated) {
						if (activated != null) {
							int length = activated.data().length;
							Tensor ret = Tensor.zeros(activated.shape());
							for (int i = 0 ; i < length; i++) {
								ret.data()[i].addi(derivative(activated.data()[i]));
							}
							return ret;
						}
						return null;
					}
				};
			case NONE:
				return new Activator() {

					@Override
					public FloatMatrix output(FloatMatrix input) {
						if (input != null)
							return input.dup();
						return null;
					}

					@Override
					public FloatMatrix derivative(FloatMatrix activated) {
						if (activated != null)
							return FloatMatrix.ones(activated.rows, activated.columns);
						return null;
					}

					@Override
					public Tensor output(Tensor input) {
						return input.dup();
					}

					@Override
					public Tensor derivative(Tensor activated) {
						if (activated != null)
							return activated.dup();
						return null;
					}
				};

			case TANH:
				return new Activator() {
					@Override
					public FloatMatrix output(FloatMatrix input) {
						if (input != null)
							return MatrixFunctions.tanh(input);
						return null;
					}

					@Override
					public FloatMatrix derivative(FloatMatrix input) {
						if (input != null)
							return FloatMatrix.ones(input.rows, input.columns).subi(MatrixFunctions.pow(input, 2));
						return null;
					}

					@Override
					public Tensor output(Tensor input) {
						if (input != null) {
							int length = input.data().length;
							Tensor ret = Tensor.zeros(input.shape());
							for (int i = 0 ; i < length; i++) {
								ret.data()[i].addi(output(input.data()[i]));
							}
							return ret;
						}
						return null;
					}

					@Override
					public Tensor derivative(Tensor activated) {
						if (activated != null) {
							int length = activated.data().length;
							Tensor ret = Tensor.zeros(activated.shape());
							for (int i = 0 ; i < length; i++) {
								ret.data()[i].addi(derivative(activated.data()[i]));
							}
							return ret;
						}
						return null;
					}
				};

			default:
				return null;
			}
		
	}
}
