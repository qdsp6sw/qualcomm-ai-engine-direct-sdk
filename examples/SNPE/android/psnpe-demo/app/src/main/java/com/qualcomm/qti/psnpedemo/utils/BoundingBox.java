/*
 * Copyright (c) 2021 - 2024 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.psnpedemo.utils;

import java.util.ArrayList;

public class BoundingBox {
    public final float x1, y1; //min point
    public final float x2, y2; //max point

    public final float score;

    public final int class_id;

    public BoundingBox(int class_id, float score, float x1, float y1, float x2, float y2) {
        assert score >= 0;
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.score = score;
        this.class_id = class_id;
    }

    public BoundingBox(float x1, float y1, float x2, float y2) {
        this.x1 = x1;
        this.y1 = y1;
        this.x2 = x2;
        this.y2 = y2;
        this.score = 0;
        this.class_id = 0;
    }

    public float area() {
        return (y2 - y1 + 1) * (x2 - x1 + 1);
    }

    public float iou(BoundingBox box) {
        float area1 = this.area();
        float area2 = box.area();
        float inter_area = intersectArea(this, box);
        return inter_area / (area1 + area2 - inter_area);
    }

    public static float intersectArea(BoundingBox b1, BoundingBox b2) {
        float inter_x1 = Math.max(b1.x1, b2.x1);
        float inter_y1 = Math.max(b1.y1, b2.y1);
        float inter_x2 = Math.min(b1.x2, b2.x2);
        float inter_y2 = Math.min(b1.y2, b2.y2);
        float inter_h = Math.max(0, inter_y2 - inter_y1 + 1);
        float inter_w = Math.max(0, inter_x2 - inter_x1 + 1);
        return inter_h * inter_w;
    }

    //debug
    public static Mat scoresToMat (ArrayList<BoundingBox>  boxes) {
        Mat m = new Mat(boxes.size());
        for (int i = 0; i < boxes.size(); ++i) {
            m.set(boxes.get(i).score, i);
        }
        return m;
    }
    public static Mat labelsToMat (ArrayList<BoundingBox>  boxes) {
        Mat m = new Mat(boxes.size());
        for (int i = 0; i < boxes.size(); ++i) {
            m.set(boxes.get(i).class_id, i);
        }
        return m;
    }
    public static Mat boxesToMat (ArrayList<BoundingBox> boxes) {
        Mat m = new Mat(boxes.size(), 4);
        for (int i = 0; i < boxes.size(); ++i) {
            m.set(boxes.get(i).y1, i, 0);
            m.set(boxes.get(i).x1, i, 1);
            m.set(boxes.get(i).y2, i, 2);
            m.set(boxes.get(i).x2, i, 3);
        }
        return m;
    }
}
