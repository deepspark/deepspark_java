package org.acl.deepspark.data;

/**
 * Created by Jaehong on 2015-10-02.
 */
public class WeightFactory {
    public static Tensor create(WeightType t, float value, int... dimW) {
        Tensor tensor = null;
        switch (t) {
            case CONSTANT:
                tensor = Tensor.ones(dimW).muli(value);
                break;

            case UNIFORM:
                tensor = Tensor.rand(dimW).subi(0.5f).muli(2.0f*value);
                break;

            case GAUSSIAN:
                tensor = Tensor.randn(dimW).muli(value);
                break;

            case XAVIER:
                tensor = Tensor.rand(dimW).subi(0.5f).muli(2.0f*value);
                break;
        }
        return tensor;
    }
}
