package com.baidu.paddle.paddlenlp.ernie_tiny;

import android.support.annotation.NonNull;

import java.util.ArrayList;
import java.util.Arrays;

public class SchemaNode {
    // Relation Extraction in native: Pass 'SchemaNode[] schema' from Java
    // SchemaNode c_node_0; // From Java schema[0] Java SchemaNode via BFS
    // SchemaNode c_node_1; // From Java schema[1] Java SchemaNode via BFS
    // predictor.SetSchema({c_node_0, c_node_1});
    // predictor.Predict({"xxx"}, &results);
    public String mName;
    public ArrayList<SchemaNode> mChildren = new ArrayList<SchemaNode>();

    public SchemaNode() {
    }

    public SchemaNode(String name) {
        mName = name;
    }

    public SchemaNode(String name, SchemaNode[] children) {
        mName = name;
        mChildren.addAll(Arrays.asList(children));
    }

    public SchemaNode(String name, ArrayList<SchemaNode> children) {
        mName = name;
        mChildren.addAll(children);
    }

    public void addChild(String schema) {
        mChildren.add(new SchemaNode(schema));
    }

    public void addChild(SchemaNode schema) {
        mChildren.add(schema);
    }

    public void addChild(String schema, @NonNull String[] children) {
        SchemaNode schemaNode = new SchemaNode(schema);
        for (String child : children) {
            schemaNode.mChildren.add(new SchemaNode(child));
        }
        mChildren.add(schemaNode);
    }

    public void addChild(String schema, ArrayList<SchemaNode> children) {
        SchemaNode schemaNode = new SchemaNode(schema);
        schemaNode.mChildren = children;
        mChildren.add(schemaNode);
    }
}
