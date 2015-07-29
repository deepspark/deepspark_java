package org.acl.deepspark.functions;

import org.acl.deepspark.nn.functions.ActivatorFactory;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * Created by Jaehong on 2015-07-29.
 */
public class ActivatorFactoryTest {
    public static void main(String[] args) {

        INDArray ones = Nd4j.ones(3, 3);

        System.out.println(ActivatorFactory.getActivator(ActivatorType.SIGMOID).output(ones));

    }


}
