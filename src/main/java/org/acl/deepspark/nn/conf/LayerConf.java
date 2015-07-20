package org.acl.deepspark.nn.conf;

import org.acl.deepspark.nn.functions.Activator;
import org.acl.deepspark.nn.functions.ActivatorType;
import org.acl.deepspark.nn.layers.LayerType;

import java.util.HashMap;

/**
 * Created by Jaehong on 2015-07-16.
 */
public class LayerConf {

    private HashMap<String, Object> layerParams;
    private LayerType type;

    public LayerConf(LayerType type) {
        this.type = type;
        layerParams = new HashMap<>();
        setActivator(ActivatorType.SIGMOID);
    }

    public void setOutputUnit(int dimOut) { layerParams.put("dimOut", dimOut); }

    public void setFilterSize(int[] filterSize) { layerParams.put("filterSize", filterSize); }

    public void setNumFilters(int numFilters) { layerParams.put("numFilters", numFilters); }

    public void setPoolingSize(int poolingSize) { layerParams.put("poolingSize", poolingSize); }

    public void setActivator(ActivatorType activator) { layerParams.put("activator", activator); }

    public int[] getInputDim() { return (int[]) layerParams.get("dimIn"); }

    public int[] getOutputDim() { return (int[]) layerParams.get("dimOut"); }

    public int[] getFilterSize() { return (int[]) layerParams.get("filterSize"); }

    public int getNumFilters() {return (int) layerParams.get("numFilters"); }

    public int getPoolingSize() { return (int) layerParams.get("poolingSize"); }

    public ActivatorType getActivator() { return (ActivatorType) layerParams.get("activator"); }

    public LayerType getType() {
        return type;
    }
}
