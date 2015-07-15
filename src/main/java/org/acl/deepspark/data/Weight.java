package org.acl.deepspark.data;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by Jaehong on 2015-07-14.
 */
public class Weight {
    private INDArray params;
    private double bias;

    public Weight(double[] data, int[] shape) {
        params = Nd4j.create(data, shape);
    }

    public Weight(int[] shape) {
        params = Nd4j.randn(shape);
    }

    public int[] getDims() {
        return params.shape();
    }

    public void setParams(INDArray params) {
        this.params = params;
    }

    public INDArray getParams() {
        return params;
    }

    public void addWeight(INDArray params) {
        this.params.addi(params);
    }

    public void subWeight(INDArray params) {
        this.params.subi(params);
    }


    public void addBias(double d) {
        bias += d;
    }

    public void subBias(double d) {
        bias -= d;
    }

    public void mmulWeight(INDArray params) {
        this.params.mmul(params);
    }

    public void mmuliWeight(INDArray params) {
        this.params.mmuli(params);
    }

    public void mulScalar(double d) {
        this.params.muli(d);
    }

    public double sum() {
        double sum = 0;
        for(int i = 0 ; i < params.length(); i++)
            sum += params.getDouble(i);
        return sum;
    }


}
