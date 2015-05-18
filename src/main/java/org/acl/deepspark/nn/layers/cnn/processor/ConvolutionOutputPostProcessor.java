
package org.acl.deepspark.nn.layers.cnn.processor;


/**
 * Used for feeding the output of a conv net in to a 2d classifier.
 * Takes the output shape of the convolution in 4d and reshapes it to a 2d
 * by using the first output as the batch size, and taking the columns via the prod operator
 * for the rest. The shape of the output is inferred by the passed in layer.
 *
 * @author Adam Gibson
 */
public class ConvolutionOutputPostProcessor {

}
