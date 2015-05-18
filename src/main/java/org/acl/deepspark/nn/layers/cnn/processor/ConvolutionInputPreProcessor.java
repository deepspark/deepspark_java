
package org.acl.deepspark.nn.layers.cnn.processor;

/**
 * A convolution input pre processor.
 * When passing things in to a convolutional net, a 4d tensor is expected of shape:
 * batch size, 1, rows, cols
 *
 * For a typical flattened dataset of images which are of:
 * batch size x rows * cols in size, this gives the equivalent transformation for a convolutional layer of:
 *
 * batch size (inferred from matrix) x 1 x rows x columns
 *
 * Note that for any output passed in, the number of columns of the passed in feature matrix must be equal to
 * rows * cols passed in to the pre processor.
 *
 * @author Adam Gibson
 */

public class ConvolutionInputPreProcessor {
    
}
