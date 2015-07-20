package org.acl.deepspark.nn.conf;

import java.util.HashMap;

import org.acl.deepspark.nn.layers.LayerType;

/**
 * Created by Jaehong on 2015-07-16.
 */
public class LayerConf {

    private HashMap<String, Object> layerParams;
    private LayerType type;

    public LayerConf(LayerType type) {
        this.type = type;
        layerParams = new HashMap<>();
    }

    public void set(String key, Object value) { layerParams.put(key, value); }
    
    public Object get(String key) { return layerParams.get(key); }
    
    public LayerType getType() {
        return type;
    }
}
