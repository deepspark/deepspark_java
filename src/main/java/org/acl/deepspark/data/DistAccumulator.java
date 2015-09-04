package org.acl.deepspark.data;

import org.apache.spark.AccumulatorParam;

/**
 * Created by Jaehong on 2015-07-31.
 */
public class DistAccumulator implements AccumulatorParam<Weight[]> {

    @Override
    public Weight[] addAccumulator(Weight[] current, Weight[] param) {
        if (param != null && current.length != param.length)
            throw new IllegalArgumentException(String.format
                    ("Number of layers mismatch; current %d, param %d", current.length, param.length));

        for (int i = 0 ; i < current.length; i++) {
            if (current[i] != null)
                current[i].addi(param[i]);
            else
                current[i] = param[i];
        }
        return current;
    }

    @Override
    public Weight[] addInPlace(Weight[] current, Weight[] param) {
        if (param != null && current.length != param.length)
            throw new IllegalArgumentException(String.format
                    ("Number of layers mismatch; current %d, param %d", current.length, param.length));

        for (int i = 0 ; i < current.length; i++) {
            if (current[i] != null)
                current[i].addi(param[i]);
            else
                current[i] = param[i];
        }
        return current;
    }

    @Override
    public Weight[] zero(Weight[] weights) {
        for (int i = 0 ; i < weights.length; i++) {
            if (weights[i] != null)
                weights[i] = new Weight(weights[i].getWeightShape(), weights[i].getBiasShape());
        }
        return weights;
    }
}
